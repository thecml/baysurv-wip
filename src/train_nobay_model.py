import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model_builder import TrainAndEvaluateModel, Predictor, make_nobay_model
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from data_loader import load_cancer_ds, prepare_cancer_ds, \
                        load_veterans_ds, prepare_veterans_ds
from sksurv.metrics import concordance_index_censored
import os
from pathlib import Path

N_EPOCHS = 10

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_veterans_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_veterans_ds(y_train, y_valid, y_test)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = CoxPHLoss()

    model = make_nobay_model(input_shape=X_train.shape[1:], output_dim=2) # Risk as CPD
    train_fn = InputFunction(X_train, t_train, e_train, drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid)
    test_fn = InputFunction(X_test, t_test, e_test)

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

    # Compute test loss
    for x, y in test_ds:
        y_event = tf.expand_dims(y["label_event"], axis=1)
        test_logits = model(x, training=False).sample()
        test_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=test_logits)
        test_loss_metric.update_state(test_loss)
    test_loss = float(test_loss_metric.result())

    # Compute average Harrell's c-index
    runs = 10
    model_nobay = np.zeros((runs, len(X_test)))
    for i in range(0, runs):
        model_nobay[i,:] = np.reshape(model.predict(X_test, verbose=0), len(X_test))
    cpd_se = np.std(model_nobay, ddof=1) / np.sqrt(np.size(model_nobay)) # standard error
    c_index_result = concordance_index_censored(e_test, t_test, model_nobay.mean(axis=0))[0] # use mean cpd for c-index
    print(f"Training completed, test loss/C-index/CPD SE: " \
          + f"{round(test_loss, 4)}/{round(c_index_result, 4)}/{round(cpd_se, 4)}")

    # Make predictions by sampling from the Gaussian posterior
    x_pred = X_test[:3]
    for rep in range(5): #Predictions for 5 runs
        print(model.predict(x_pred, verbose=0)[0:3].T)

    # Save model weights
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    model.save_weights(f'{root_dir}/models/nobay/')