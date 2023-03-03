import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from sksurv.metrics import concordance_index_censored
from tools.model_builder import make_mc_dropout_model
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
from tools.preprocessor import Preprocessor
from tools import data_loader
from tools.model_builder import Trainer, Predictor
from tools.model_builder import make_baseline_model
from utility.risk import InputFunction
from utility.metrics import CindexMetric
from utility.loss import CoxPHLoss

tfd = tfp.distributions
tfb = tfp.bijectors

N_EPOCHS = 10

if __name__ == "__main__":
    # Load data
    dl = data_loader.CancerDataLoader().load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = dl.prepare_data()
    t_train, t_valid, t_test, e_train, e_valid, e_test = dl.make_time_event_split(y_train, y_valid, y_test)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = CoxPHLoss()

    # MC Dropout
    model = make_mc_dropout_model(input_shape=X_train.shape[1:],
                                  output_dim=2)
    
    train_fn = InputFunction(X_train, t_train, e_train, drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid, drop_last=True) #for testing
    test_fn = InputFunction(X_test, t_test, e_test, drop_last=True) #for testing

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    test_loss_metric = tf.keras.metrics.Mean(name="test_loss")

    val_cindex_metric = CindexMetric()
    train_loss_scores = list()
    val_loss_scores, val_cindex_scores = list(), list()

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
        
        # Run validation loop
        train_loss_metric.reset_states()
        val_cindex_metric.reset_states()
        for x_val, y_val in val_ds:
            y_event = tf.expand_dims(y_val["label_event"], axis=1)
            val_logits = model(x_val, training=False).sample()
            val_loss = loss_fn(y_true=[y_event, y_val["label_riskset"]], y_pred=val_logits)
            
            # Update val metrics
            val_loss_metric.update_state(val_loss)
            val_cindex_metric.update_state(y_val, val_logits)
        
        val_loss = val_loss_metric.result()
        val_cindex = val_cindex_metric.result()
        
        val_loss_scores.append(float(val_loss))
        val_cindex_scores.append(val_cindex['cindex'])
        
        print(f"Training loss = {train_loss:.4f} - " + \
              f"Validation loss = {val_loss:.4f} - " + \
              f"C-index = {val_cindex['cindex']:.4f}")
    
    # Compute test loss
    for x, y in test_ds:
        y_event = tf.expand_dims(y["label_event"], axis=1)
        test_logits = model(x, training=False).sample()
        test_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=test_logits)
        test_loss_metric.update_state(test_loss)
    test_loss = float(test_loss_metric.result())
    
    # Compute average Harrell's c-index
    runs = 100
    model_cpd = np.zeros((runs, len(X_test)))
    for i in range(0, runs):
        model_cpd[i,:] = np.reshape(model.predict(X_test, verbose=0), len(X_test))
    cpd_se = np.std(model_cpd, ddof=1) / np.sqrt(np.size(model_cpd)) # standard error
    c_index_result = concordance_index_censored(e_test, t_test, model_cpd.mean(axis=0))[0] # use mean cpd for c-index
    print(f"Training completed, test loss/C-index/CPD SE: " \
          + f"{round(test_loss, 4)}/{round(c_index_result, 4)}/{round(cpd_se, 4)}")
    
    # Save model weights
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    model.save_weights(f'{root_dir}/models/mc_dropout/')
        
    # Get MC dropout predictions
    #x_pred = X_test[:3]
    #runs = 10
    #mc_cpd = np.zeros((runs,len(x_pred)))
    #for i in range(0,runs):
    #    mc_cpd[i,:] = np.reshape(model.predict(x_pred, verbose=0), len(x_pred))
    #print(mc_cpd)
    
    # No dropout at test time
    #model_mc_pred = K.function([model.input, tf.constant(K.learning_phase())], [model.output])
    #for _ in range(5): #Predictions for 5 runs
    #    print(model_mc_pred([X_test[:3],0])[0])
    
   # Dropout at test time
    #for _ in range(5):
    #    print(model_mc_pred([X_test[:3],1])[0])