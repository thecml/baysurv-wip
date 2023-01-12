import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_risk_model import TrainAndEvaluateModel, Predictor
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
from data_loader import load_veterans_ds, load_cancer_ds, load_aids_ds, load_nhanes_ds # datasets
from data_loader import prepare_veterans_ds, prepare_cancer_ds, \
                        prepare_aids_ds, prepare_nhanes_ds # prepare funcs
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

N_EPOCHS = 100

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_nhanes_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_nhanes_ds(y_train, y_valid, y_test)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = tf.keras.layers.Dense(30, activation="relu")(inputs)
    hidden1= tf.keras.layers.Dropout(0.5)(hidden1)
    hidden1 = tf.keras.layers.BatchNormalization()(hidden1)
    output = tf.keras.layers.Dense(1, activation="linear")(hidden1)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    train_fn = InputFunction(X_train, t_train, e_train,
                             drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid, drop_last=True) # for testing
    test_fn = InputFunction(X_test, t_test, e_test, drop_last=True) #for testing

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = CoxPHLoss()

    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    test_loss_metric = tf.keras.metrics.Mean(name="test_loss")

    train_cindex_metric = CindexMetric()
    val_cindex_metric = CindexMetric()

    train_loss_scores, train_cindex_scores = list(), list()
    valid_loss_scores, valid_cindex_scores = list(), list()

    train_ds = train_fn()
    val_ds = val_fn()
    test_ds = test_fn()

    for epoch in range(N_EPOCHS):
        #Training step
        train_cindex_metric.reset_states()
        for x, y in train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                train_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                train_cindex_metric.update_state(y, logits)

            with tf.name_scope("gradients"):
                grads = tape.gradient(train_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_loss_metric.update_state(train_loss)

        # Save metrics
        mean_loss = train_loss_metric.result()
        mean_cindex = train_cindex_metric.result()
        train_loss_scores.append(float(mean_loss))
        train_cindex_scores.append(mean_cindex['cindex'])

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
            latest_train_loss = train_loss_scores[-1]
            latest_train_cindex = train_cindex_scores[-1]
            print(f"Train loss = {latest_train_loss:.4f}, Train CI = {latest_train_cindex:.4f}, " \
                  + f"Valid loss = {val_loss:.4f}, Valid CI = {val_cindex['cindex']:.4f}")

    # Compute test loss
    for x, y in test_ds:
        y_event = tf.expand_dims(y["label_event"], axis=1)
        test_logits = model(x, training=False)
        test_loss = loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=test_logits)
        test_loss_metric.update_state(test_loss)
    test_loss = float(test_loss_metric.result())

    # Compute Harrell's c-index
    sample_predictions = model.predict(X_test).reshape(-1)
    c_index_result = concordance_index_censored(e_test.astype(bool), t_test, sample_predictions)[0]
    print(f"Training completed, test loss/C-index: {round(test_loss, 4)}/{round(c_index_result, 4)}")