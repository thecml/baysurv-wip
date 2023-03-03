import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tools.model_builder import Trainer, Predictor, make_nobay_model
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from tools import data_loader
from sksurv.metrics import concordance_index_censored
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tools.model_builder import Trainer, Predictor
from utility.risk import InputFunction
from utility.loss import CoxPHLossLA, CoxPHLoss
from utility.metrics import CindexMetric
import random

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

N_EPOCHS = 10

if __name__ == "__main__":
    # Load data
    dl = data_loader.SupportDataLoader().load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = dl.prepare_data()
    t_train, t_valid, t_test, e_train, e_valid, e_test = dl.make_time_event_split(y_train, y_valid, y_test)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = CoxPHLossLA()

    model = make_nobay_model(input_shape=X_train.shape[1:], output_dim=2) # Risk as CPD
    train_fn = InputFunction(X_train, t_train, e_train, drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid)
    test_fn = InputFunction(X_test, t_test, e_test)

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
        # Run training loop
        train_loss_metric.reset_states()
        for x, y in train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                y_pred_dist = model(x, training=True)
                train_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=y_pred_dist)
                                
            with tf.name_scope("gradients"):
                grads = tape.gradient(train_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_loss_metric.update_state(train_loss)

        # Save metrics
        train_loss = train_loss_metric.result()
        train_loss_scores.append(float(train_loss))
        
        # Run validation loop
        val_loss_metric.reset_states()
        val_cindex_metric.reset_states()
        val_var_mean_metric = list()
        
        for x_val, y_val in val_ds:
            y_event = tf.expand_dims(y_val["label_event"], axis=1)
            y_pred_dist = model(x_val, training=False)
            val_loss = loss_fn(y_true=[y_event, y_val["label_riskset"]], y_pred=y_pred_dist)
            
            # Compute sum of variance of val CPD
            runs = 100
            y_pred_cpd = np.zeros((runs, y_pred_dist.batch_shape[0]), dtype=np.float32)
            for i in range(0, runs):
                y_pred_cpd[i,:] = np.reshape(y_pred_dist.sample(), y_pred_dist.batch_shape[0])
            sum_of_val_var = tf.reduce_sum(tf.math.reduce_variance(y_pred_cpd, axis=0))
            val_var_mean_metric.append(sum_of_val_var)
                  
            val_loss_metric.update_state(val_loss)
            val_cindex_metric.update_state(y_val, y_pred_dist)
            
        val_loss = val_loss_metric.result()
        val_cindex = val_cindex_metric.result()
        
        val_loss_scores.append(float(val_loss))
        val_cindex_scores.append(val_cindex['cindex'])
        val_var_mean = tf.reduce_mean(val_var_mean_metric).numpy()
            
        print(f"Training loss = {train_loss:.4f}, Validation: loss = {val_loss:.4f}, " \
             + f"C-index = {val_cindex['cindex']:.4f}, Variance mean = {val_var_mean:.4f}")
              
    # Compute test loss
    for x, y in test_ds:
        y_event = tf.expand_dims(y["label_event"], axis=1)
        y_pred_dist = model(x, training=False)
        test_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=y_pred_dist)
        test_loss_metric.update_state(test_loss)
    test_loss = float(test_loss_metric.result())

    # Compute average Harrell's c-index
    runs = 100
    model_nobay = np.zeros((runs, len(X_test)))
    for i in range(0, runs):
        model_nobay[i,:] = np.reshape(model.predict(X_test, verbose=0), len(X_test))
    cpd_se = np.std(model_nobay, ddof=1) / np.sqrt(len(X_test)) # SE of predictions
    predictions = model_nobay.mean(axis=0)
    c_index_result = concordance_index_censored(e_test.astype(bool), t_test, predictions)[0] # use mean cpd for c-index
    print(f"Training completed, test loss/C-index/CPD SE: " \
          + f"{round(test_loss, 4)}/{round(c_index_result, 4)}/{round(cpd_se, 4)}")

    # Save model weights
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    model.save_weights(f'{root_dir}/models/nobay/')