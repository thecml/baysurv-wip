import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utility.risk import InputFunction
from utility.metrics import CindexMetric
from utility.loss import CoxPHLoss
import tensorflow_probability as tfp

from tools.model_builder import make_vi_model
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
from tools import data_loader

tfd = tfp.distributions
tfb = tfp.bijectors

N_EPOCHS = 50

if __name__ == "__main__":
    dl = data_loader.GbsgDataLoader().load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = dl.prepare_data()
    t_train, t_valid, t_test, e_train, e_valid, e_test = dl.make_time_event_split(y_train, y_valid, y_test)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = CoxPHLoss()

    # VI
    model = make_vi_model(n_train_samples=X_train.shape[0],
                          input_shape=X_train.shape[1:],
                          output_dim=2) # output_dim=1 to only model epistemic uncertainty

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
        kl_loss = list()
        for x, y in train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                logits = model(x, training=True).sample()
                train_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                train_loss = train_loss + tf.reduce_mean(model.losses) # CoxPHLoss + KL-divergence
                kl_loss.append(tf.reduce_mean(model.losses))

            with tf.name_scope("gradients"):
                grads = tape.gradient(train_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_loss_metric.update_state(train_loss)

        # Save metrics
        mean_loss = train_loss_metric.result()
        train_loss_scores.append(float(mean_loss))
        train_loss_metric.reset_states()
        val_cindex_metric.reset_states()

        # Run validation loop
        val_loss_metric.reset_states()
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
    model.save_weights(f'{root_dir}/models/vi/')