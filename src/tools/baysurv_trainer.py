import tensorflow as tf
import numpy as np
from utility.metrics import CindexMetric, CtdMetric, IbsMetric, InbllMetric
from utility.survival import convert_to_structured
from time import time
from utility.loss import CoxPHLoss, CoxPHLossLLA

class Trainer:
    def __init__(self, model, model_name, train_dataset, valid_dataset,
                 test_dataset, optimizer, loss_function, num_epochs,
                 event_times, early_stop, patience):
        self.num_epochs = num_epochs
        self.model = model
        self.model_name = model_name

        self.train_ds = train_dataset
        self.valid_ds = valid_dataset
        self.test_ds = test_dataset

        self.optimizer = optimizer
        self.loss_fn = loss_function

        self.train_loss_scores = list()
        self.valid_loss_scores, self.valid_ci_scores = list(), list()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_ctd_metric = CtdMetric(event_times)
        self.train_ibs_metric = IbsMetric(event_times)
        self.train_inbll_metric = InbllMetric(event_times)
        
        self.valid_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        
        self.test_loss_metric = tf.keras.metrics.Mean(name="test_loss")
        self.test_ctd_metric = CtdMetric(event_times)
        self.test_ibs_metric = IbsMetric(event_times)
        self.test_inbll_metric = InbllMetric(event_times)

        self.train_loss_scores, self.train_inbll_scores = list(), list()
        self.train_ctd_scores, self.train_ibs_scores = list(), list()
        
        self.valid_loss_scores = list()
        
        self.test_loss_scores, self.test_inbll_scores = list(), list()
        self.test_ctd_scores, self.test_ibs_scores = list(), list()

        self.train_times, self.test_times = list(), list()
        
        self.test_variance = list()
        
        self.early_stop = early_stop
        self.patience = patience
        
        self.best_val_nll = np.inf
        self.best_ep = -1

    def train_and_evaluate(self):
        stop_training = False
        for epoch in range(1, self.num_epochs+1):
            self.train(epoch)
            if self.valid_ds is not None:
                stop_training = self.validate(epoch)
            if self.test_ds is not None:
                self.test()
            if stop_training:
                self.cleanup()
                break
            self.cleanup()

    def train(self, epoch):
        train_start_time = time()
        for x, y in self.train_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            with tf.GradientTape() as tape:
                if self.model_name in ["MLP-ALEA", "VI", "VI-EPI", "MCD"]:
                    runs = 100
                    logits_cpd = tf.zeros((runs, y_event.shape[0]), dtype=np.float32)
                    output_list = []
                    tensor_shape = logits_cpd.get_shape()
                    for i in range(tensor_shape[0]):
                        y_pred = self.model(x, training=True)
                        if self.model_name in ["MLP-ALEA", "VI", "MCD"]:
                            output_list.append(tf.reshape(y_pred.sample(), y_pred.shape[0]))
                        else:
                            output_list.append(tf.reshape(y_pred, y_pred.shape[0]))
                    logits_cpd = tf.stack(output_list)
                    cox_loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_cpd)
                    logits = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
                    loss = cox_loss + tf.reduce_mean(self.model.losses) # CoxPHLoss + KL-divergence
                    self.train_loss_metric.update_state(cox_loss)
                else:
                    logits = self.model(x, training=True)
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                    self.train_loss_metric.update_state(loss)
                y_train = convert_to_structured(y["label_time"], y["label_event"])

                # CTD
                self.train_ctd_metric.update_train_state(y_train)
                self.train_ctd_metric.update_train_pred(logits)
                self.test_ctd_metric.update_train_state(y_train)
                self.test_ctd_metric.update_train_pred(logits)
                
                # IBS
                self.train_ibs_metric.update_train_state(y_train)
                self.train_ibs_metric.update_train_pred(logits)
                self.test_ibs_metric.update_train_state(y_train)
                self.test_ibs_metric.update_train_pred(logits)
                
                # INBLL
                self.train_inbll_metric.update_train_state(y_train)
                self.train_inbll_metric.update_train_pred(logits)
                self.test_inbll_metric.update_train_state(y_train)
                self.test_inbll_metric.update_train_pred(logits)
                
            with tf.name_scope("gradients"):
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        print(f"Completed {self.model_name} epoch {epoch}/{self.num_epochs}")
        total_train_time = time() - train_start_time

        epoch_loss = self.train_loss_metric.result()
        epoch_ctd = self.train_ctd_metric.result()
        epoch_ibs = self.train_ibs_metric.result()
        epoch_inbll = self.train_inbll_metric.result()

        self.train_loss_scores.append(float(epoch_loss))
        self.train_ctd_scores.append(float(epoch_ctd))
        self.train_ibs_scores.append(float(epoch_ibs))
        self.train_inbll_scores.append(float(epoch_inbll))

        self.train_times.append(float(total_train_time))

    def validate(self, epoch):
        stop_training = False
        for x, y in self.valid_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            if self.model_name in ["MLP-ALEA", "VI", "VI-EPI", "MCD"]:
                runs = 100
                logits_cpd = np.zeros((runs, len(x)), dtype=np.float32)
                for i in range(0, runs):
                    if self.model_name in ["MLP-ALEA", "VI", "MCD"]:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False).sample(), len(x))
                    else:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False), len(x))
                logits_mean = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
                if isinstance(self.loss_fn, CoxPHLoss):
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                else:
                    logits = self.model(x, training=False)
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_cpd)
                self.valid_loss_metric.update_state(loss)
            else:
                logits = self.model(x, training=False)
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.valid_loss_metric.update_state(loss)

        epoch_valid_loss = self.valid_loss_metric.result()
        self.valid_loss_scores.append(float(epoch_valid_loss))
        
        # Early stopping
        if self.early_stop:
            print(f'Best Val NLL: {self.best_val_nll}, epoch Val NNL: {epoch_valid_loss}')
            if self.best_val_nll > epoch_valid_loss:
                self.best_val_nll = epoch_valid_loss
                self.best_ep = epoch
            if (epoch - self.best_ep) > self.patience:
                print(f"Validation loss converges at {self.best_ep}th epoch.")
                stop_training = True
            else:
                stop_training = False
                
        return stop_training

    def test(self):
        test_start_time = time()
        batch_stds = list()
        for x, y in self.test_ds:
            y_event = tf.expand_dims(y["label_event"], axis=1)
            if self.model_name in ["MLP-ALEA", "VI", "VI-EPI", "MCD"]:
                runs = 100
                logits_cpd = np.zeros((runs, len(x)), dtype=np.float32)
                for i in range(0, runs):
                    if self.model_name in ["MLP-ALEA", "VI", "MCD"]:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False).sample(), len(x))
                    else:
                        logits_cpd[i,:] = np.reshape(self.model(x, training=False), len(x))
                logits_mean = tf.transpose(tf.reduce_mean(logits_cpd, axis=0, keepdims=True))
                mean_std = np.mean(tf.math.reduce_std(logits_cpd, axis=0, keepdims=True))
                batch_stds.append(mean_std)
                
                #logits = self.model(x, training=False)
                if isinstance(self.loss_fn, CoxPHLoss):
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_mean)
                else:
                    loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits_cpd)
                
                self.test_loss_metric.update_state(loss)
                y_test = convert_to_structured(y["label_time"], y["label_event"])
                self.test_ctd_metric.update_test_state(y_test)
                self.test_ctd_metric.update_test_pred(logits_mean)
                self.test_ibs_metric.update_test_state(y_test)
                self.test_ibs_metric.update_test_pred(logits_mean)
                self.test_inbll_metric.update_test_state(y_test)
                self.test_inbll_metric.update_test_pred(logits_mean)
                
            else:
                logits = self.model(x, training=False)
                loss = self.loss_fn(y_true=[y_event, y["label_riskset"]], y_pred=logits)
                self.test_loss_metric.update_state(loss)
                y_test = convert_to_structured(y["label_time"], y["label_event"])
                self.test_ctd_metric.update_test_state(y_test)
                self.test_ctd_metric.update_test_pred(logits)
                self.test_ibs_metric.update_test_state(y_test)
                self.test_ibs_metric.update_test_pred(logits)
                self.test_inbll_metric.update_test_state(y_test)
                self.test_inbll_metric.update_test_pred(logits)
         
        total_test_time = time() - test_start_time
        
        # Std
        if len(batch_stds) > 0:
            model_var = float(np.mean(batch_stds) ** 2)
            print(f"Model test variance: {model_var}")
            self.test_variance.append(model_var)

        epoch_loss = self.test_loss_metric.result()
        epoch_ctd = self.test_ctd_metric.result()
        epoch_ibs = self.test_ibs_metric.result()
        epoch_inbll = self.test_inbll_metric.result()

        self.test_loss_scores.append(float(epoch_loss))
        self.test_ctd_scores.append(float(epoch_ctd))
        self.test_ibs_scores.append(float(epoch_ibs))
        self.test_inbll_scores.append(float(epoch_inbll))
        self.test_times.append(float(total_test_time))
    
    def cleanup(self):
        self.train_loss_metric.reset_states()
        self.train_ctd_metric.reset_states()
        self.train_ibs_metric.reset_states()
        self.train_inbll_metric.reset_states()
        self.valid_loss_metric.reset_states()
        self.test_loss_metric.reset_states()
        self.test_ctd_metric.reset_states()
        self.test_ibs_metric.reset_states()
        self.test_inbll_metric.reset_states()