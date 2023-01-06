import tensorflow as tf
import numpy as np
from utility import CoxPHLoss, CindexMetric

class Predictor:
    def __init__(self, model):
        self.model = model
    def predict(self, dataset):
        risk_scores = []
        for batch in dataset:
            pred = self.model(batch, training=False)
            risk_scores.append(pred.numpy())
        return np.row_stack(risk_scores)

class TrainAndEvaluateModel:
    """
    Wrapper for a TensorFlow risk model that can predict survival probabilities.
    See: https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/
    """
    def __init__(self, model, train_dataset, eval_dataset,
                learning_rate, num_epochs):
        self.num_epochs = num_epochs

        self.model = model

        self.train_ds = train_dataset
        self.val_ds = eval_dataset

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()

        self.train_loss_scores = list()
        self.valid_loss_scores, self.valid_cindex_scores = list(), list()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.val_cindex_metric = CindexMetric()

    @tf.function
    def train_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            train_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=logits)

        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss, logits

    def train_and_evaluate(self):
        step = tf.Variable(0, dtype=tf.int64)
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.evaluate(epoch)

    def train_one_epoch(self, epoch_counter):
        for x, y in self.train_ds:
            train_loss, logits = self.train_one_step(
                x, y["label_event"], y["label_riskset"])

            # Update training metric.
            self.train_loss_metric.update_state(train_loss)

        # Save metrics
        mean_loss = self.train_loss_metric.result()
        self.train_loss_scores.append(float(mean_loss))
        # Reset training metrics
        self.train_loss_metric.reset_states()

    @tf.function
    def evaluate_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        val_logits = self.model(x, training=False)
        val_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
        return val_loss, val_logits

    def evaluate(self, step_counter):
        self.val_cindex_metric.reset_states()

        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(
                x_val, y_val["label_event"], y_val["label_riskset"])

            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_cindex_metric.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()
        self.val_loss_metric.reset_states()

        self.valid_loss_scores.append(float(val_loss))

        val_cindex = self.val_cindex_metric.result()
        self.valid_cindex_scores.append(val_cindex['cindex'])
        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")