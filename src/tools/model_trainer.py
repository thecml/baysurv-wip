import tensorflow as tf
from utility.metrics import CindexMetric
import numpy as np
from sksurv.metrics import concordance_index_censored

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset,
                 test_dataset, optimizer, loss_function, num_epochs):
        self.num_epochs = num_epochs
        self.model = model

        self.train_ds = train_dataset
        self.valid_ds = valid_dataset
        self.test_ds = test_dataset

        self.optimizer = optimizer
        self.loss_fn = loss_function

        self.train_loss_scores = list()
        self.valid_loss_scores, self.valid_ci_scores = list(), list()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_cindex_metric = CindexMetric()

        self.valid_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.valid_cindex_metric = CindexMetric()

        self.train_loss_scores, self.train_ci_scores = list(), list()
        self.valid_loss_scores, self.valid_ci_scores = list(), list()

    def train_and_evaluate(self):
        for _ in range(self.num_epochs):
            self.train()
            self.validate()
            if self.test_ds is not None:
                self.test()
            self.cleanup()

    def train(self):
        for x, y in self.train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.train_loss_metric.update_state(loss)
                self.train_cindex_metric.update_state(y, logits)

            with tf.name_scope("gradients"):
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        epoch_loss = self.train_loss_metric.result()
        epoch_ci = self.train_cindex_metric.result()['cindex']
        self.train_loss_scores.append(float(epoch_loss))
        self.train_ci_scores.append(float(epoch_ci))

    def validate(self):
        for x, y in self.valid_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            logits = self.model(x, training=False)
            loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
            self.valid_loss_metric.update_state(loss)
            self.valid_cindex_metric.update_state(y, logits)

        epoch_valid_loss = self.valid_loss_metric.result()
        epoch_valid_ci = self.valid_cindex_metric.result()['cindex']
        self.valid_loss_scores.append(float(epoch_valid_loss))
        self.valid_ci_scores.append(float(epoch_valid_ci))

    def cleanup(self):
        self.train_loss_metric.reset_states()
        self.train_cindex_metric.reset_states()
        self.valid_loss_metric.reset_states()
        self.valid_cindex_metric.reset_states()

    def test(self):
        risk_scores, e_test, t_test = [], [], []
        for x, y in self.test_ds:
            e_test.extend(y['label_event'].numpy())
            t_test.extend(y['label_time'].numpy())
            y_pred = self.model(x, training=False)
            risk_scores.append(y_pred.numpy())

        # Compute Harrell's C-index
        test_predictions = np.row_stack(risk_scores).reshape(-1)
        e_test = np.array(e_test, dtype=bool)
        t_test = np.array(t_test)
        self.test_ci = concordance_index_censored(e_test, t_test, test_predictions)[0]


