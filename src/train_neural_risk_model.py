import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from neural_risk_model import TrainAndEvaluateModel, Predictor
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.datasets import load_veterans_lung_cancer
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

@tf.function
def evaluate_one_step(model, loss_fn, x, y_event, y_riskset):
    y_event = tf.expand_dims(y_event, axis=1)
    val_logits = model(x, training=False)
    val_loss = loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
    return val_loss, val_logits

@tf.function
def train_one_step(model, loss_fn, optimizer, x, y_event, y_riskset):
    y_event = tf.expand_dims(y_event, axis=1)
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        train_loss = loss_fn(y_true=[y_event, y_riskset], y_pred=logits)

    with tf.name_scope("gradients"):
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return train_loss, logits

if __name__ == "__main__":
    data_x, data_y = load_veterans_lung_cancer()
    data_x = data_x[['Age_in_years', 'Karnofsky_score', 'Months_from_Diagnosis']] # use only num vars

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
                                                        test_size=0.3, random_state=0)

    time_train = np.array(y_train['Survival_in_days'])
    event_train = np.array(y_train['Status'])
    time_test = np.array(y_test['Survival_in_days'])
    event_test = np.array(y_test['Status'])
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    def normal_sp(params):
        return tfd.Normal(loc=params[:,0:1],
                            scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable

    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    hidden = tf.keras.layers.Dense(30,activation="relu")(inputs)
    params = tf.keras.layers.Dense(2)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp,
                                         convert_to_tensor_fn=tfp.distributions.Distribution.sample)(params) 
    
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    
    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Dense(30, activation='relu',
    #                               input_shape=X_train.shape[1:]))
    #model.add(tf.keras.layers.Dense(1, activation='linear'))

    train_fn = InputFunction(X_train, time_train, event_train,
                             drop_last=True, shuffle=True)
    eval_fn = InputFunction(X_test, time_test, event_test)

    num_epochs = 50
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = CoxPHLoss()
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
    val_cindex_metric = CindexMetric()
    train_loss_scores = list()
    valid_loss_scores, valid_cindex_scores = list(), list()
    
    train_ds = train_fn()
    val_ds = eval_fn()
    
    for epoch in range(num_epochs):
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

        for x_val, y_val in val_ds:
            y_event = tf.expand_dims(y_val["label_event"], axis=1)
            val_logits = model(x, training=False)
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
    
    '''
    train_loss_scores = trainer.train_loss_scores
    valid_loss_scores = trainer.valid_loss_scores
    valid_cindex_scores = trainer.valid_cindex_scores

    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_loss_scores, color='blue', label='Training loss')
    plt.plot(epochs_range, valid_loss_scores, color='red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, valid_cindex_scores, color='red', label='Validation cindex')
    plt.title('Validation cindex')
    plt.xlabel('Epochs')
    plt.ylabel('Cindex')
    plt.legend()

    train_pred_fn = tf.data.Dataset.from_tensor_slices(X_train[..., np.newaxis]).batch(64)

    predictor = Predictor(model)
    train_predictions = predictor.predict(train_pred_fn)

    breslow = BreslowEstimator().fit(train_predictions, event_train, time_train)

    sample = train_test_split(X_test, y_test, event_test, time_test,
                              test_size=10, stratify=event_test, random_state=0)
    x_sample, y_sample, event_sample, time_sample = sample[1::2]

    sample_pred_ds = tf.data.Dataset.from_tensor_slices(
        x_sample[..., np.newaxis]).batch(64)
    sample_predictions = predictor.predict(sample_pred_ds)

    sample_surv_fn = breslow.get_survival_function(sample_predictions)

    styles = ('-', '--')
    plt.figure(figsize=(6, 4.5))
    for surv_fn, lbl in zip(sample_surv_fn, event_sample):
        plt.step(surv_fn.x, surv_fn.y, where="post",
                 linestyle=styles[int(lbl)])

    plt.ylim(0, 1)
    plt.ylabel("Probability of survival $P(T > t)$")
    plt.xlabel("Time $t$")
    plt.legend()
    plt.grid()
    plt.show()
    '''