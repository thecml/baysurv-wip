import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_risk_model import TrainAndEvaluateModel, Predictor
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
from data_loader import load_veterans_ds, load_cancer_ds, prepare_cancer_ds
from sksurv.metrics import concordance_index_censored

N_EPOCHS = 100

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_cancer_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_cancer_ds(y_train, y_valid, y_test)
    
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)

    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    hidden = tf.keras.layers.Dense(30, activation="relu")(inputs)
    output = tf.keras.layers.Dense(1, activation="linear")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    train_fn = InputFunction(X_train, t_train, e_train,
                             drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid)
    test_fn = InputFunction(X_test, t_test, e_test)

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = CoxPHLoss()
    
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
                logits = tf.convert_to_tensor(model(x, training=True))
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
            for x_val, y_val in val_ds:
                y_event = tf.expand_dims(y_val["label_event"], axis=1)
                val_logits = model(x_val, training=False)
                val_loss = loss_fn(y_true=[y_event, y_val["label_riskset"]], y_pred=val_logits)

                # Update val metrics
                val_loss_metric.update_state(val_loss)
                val_cindex_metric.update_state(y_val, val_logits)

            val_loss = val_loss_metric.result()
            val_loss_metric.reset_states()

            valid_loss_scores.append(float(val_loss))

            val_cindex = val_cindex_metric.result()
            valid_cindex_scores.append(val_cindex['cindex'])
            print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")
    
    # Compute test loss
    for x, y in test_ds:
        y_event = tf.expand_dims(y["label_event"], axis=1)
        test_logits = model(x, training=False)
        test_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=test_logits)
        test_loss_metric.update_state(test_loss)
    test_loss = float(test_loss_metric.result())
    
    # Compute Harrell's c-index
    sample_predictions = model.predict(X_test).reshape(-1)
    c_index_result = concordance_index_censored(e_test, t_test, sample_predictions)[0]
    print(f"Training completed, test loss/C-index: {round(test_loss, 4)}/{round(c_index_result, 4)}")