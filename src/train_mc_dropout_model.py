import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_risk_model import TrainAndEvaluateModel, Predictor
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from data_loader import load_cancer_ds, prepare_cancer_ds, load_nhanes_ds, prepare_nhanes_ds
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
    
tfd = tfp.distributions
tfb = tfp.bijectors

N_EPOCHS = 10

if __name__ == "__main__":
    # Load data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_nhanes_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_nhanes_ds(y_train, y_valid, y_test)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
   
    # MC Dropout
    def normal_exp(params):
        return tfd.Normal(loc=params[:,0:1], scale=tf.math.exp(params[:,1:2]))# both parameters are learnable
    
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = CoxPHLoss()

    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = tf.keras.layers.Dense(30, activation="relu")(inputs)
    hidden1= tf.keras.layers.Dropout(0.5)(hidden1, training=True) # run dropout in testing too
    params = tf.keras.layers.Dense(2)(hidden1)
    dist = tfp.layers.DistributionLambda(normal_exp, name='normal_exp')(params) 
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    
    train_fn = InputFunction(X_train, t_train, e_train, drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid, drop_last=True) #for testing
    test_fn = InputFunction(X_test, t_test, e_test, drop_last=True) #for testing

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    test_loss_metric = tf.keras.metrics.Mean(name="test_loss")

    val_cindex_metric = CindexMetric()
    train_loss_scores = list()
    valid_loss_scores, valid_cindex_scores = list(), list()

    train_ds = train_fn()
    val_ds = val_fn()
    test_ds = test_fn()

    for epoch in range(N_EPOCHS):
        #Training step
        for x, y in train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                logits = model(x, training=True).sample()
                train_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)

            with tf.name_scope("gradients"):
                grads = tape.gradient(train_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_loss_metric.update_state(train_loss)

        # Save metrics
        mean_loss = train_loss_metric.result()
        train_loss_scores.append(float(mean_loss))

        # Reset training metrics
        train_loss_metric.reset_states()

        # Evaluate step
        val_cindex_metric.reset_states()

        if epoch % 10 == 0:
            cpd_total_loss, cpd_total_logits = list(), list()
            for x_val, y_val in val_ds:
                runs = 100
                cpd_loss, cpd_logits = list(), list()
                for i in range(0, runs):
                    y_event = tf.expand_dims(y_val["label_event"], axis=1)
                    val_logits = model(x_val, training=False).sample()
                    val_loss = loss_fn(y_true=[y_event, y_val["label_riskset"]], y_pred=val_logits)
                    cpd_loss.append(val_loss)
                    cpd_logits.append(val_logits)

                # Update val metrics
                val_loss_metric.update_state(np.mean(cpd_loss))
                val_cindex_metric.update_state(y_val, np.mean(cpd_logits, axis=0))

                cpd_total_loss.append(np.std(cpd_loss))
                cpd_total_logits.append(np.std(cpd_logits, axis=0))

            val_loss = val_loss_metric.result()
            val_loss_metric.reset_states()

            valid_loss_scores.append(float(val_loss))

            val_cindex = val_cindex_metric.result()
            valid_cindex_scores.append(val_cindex['cindex'])

            std_val_loss, std_val_logits = np.mean(cpd_total_loss), np.mean(cpd_total_logits)

            print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}, " \
                + f"std val loss = {std_val_loss:.4f}, std val logits = {std_val_logits:.4f}")

    # Get MC dropout predictions
    x_pred = X_test[:3]
    runs = 10
    mc_cpd = np.zeros((runs,len(x_pred)))
    for i in range(0,runs):
        mc_cpd[i,:] = np.reshape(model.predict(x_pred, verbose=0), len(x_pred))
    print(mc_cpd)
    
    # No dropout at test time
    #model_mc_pred = K.function([model.input, tf.constant(K.learning_phase())], [model.output])
    #for _ in range(5): #Predictions for 5 runs
    #    print(model_mc_pred([X_test[:3],0])[0])
    
   # Dropout at test time
    #for _ in range(5):
    #    print(model_mc_pred([X_test[:3],1])[0])