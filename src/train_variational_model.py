import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model_builder import TrainAndEvaluateModel, Predictor
from utility import InputFunction, CindexMetric, CoxPHLoss
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
import tensorflow_probability as tfp
from data_loader import load_cancer_ds, prepare_cancer_ds
from sksurv.metrics import concordance_index_censored

tfd = tfp.distributions
tfb = tfp.bijectors

N_EPOCHS = 200
PLOT_SURV_FUNCS = False

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_cancer_ds()
    t_train, t_valid, t_test, e_train, e_valid, e_test  = prepare_cancer_ds(y_train, y_valid, y_test)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    loss_fn = CoxPHLoss()

    def normal_sp(params):
        return tfd.Normal(loc=params[:,0:1],
                          scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable
    def normal(params):
        return tfd.Normal(loc=params, scale=1)

    def NLL(y, distr):
      return -distr.log_prob(y)

    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] * 1.0)
    bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] * 1.0)

    # VI
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    hidden = tfp.layers.DenseFlipout(20,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn,activation="relu")(inputs)
    hidden = tfp.layers.DenseFlipout(50,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn,activation="relu")(hidden)
    hidden = tfp.layers.DenseFlipout(20,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn,activation="relu")(hidden)
    params = tfp.layers.DenseFlipout(2,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)

    '''
    inputs = tf.keras.layers.Input(shape=X_train.shape[1:])
    hidden = tf.keras.layers.Dense(30, activation="relu")(inputs)
    params = tf.keras.layers.Dense(2)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params)
    model = tf.keras.Model(inputs=inputs, outputs=dist)
    '''

    train_fn = InputFunction(X_train, t_train, e_train, drop_last=True, shuffle=True)
    val_fn = InputFunction(X_valid, t_valid, e_valid, drop_last=True) #for testing
    test_fn = InputFunction(X_test, t_test, e_test, drop_last=True) #for testing

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
        print(f"Mean KL loss: {np.mean(kl_loss)}")
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
    model_cpd = np.zeros((runs, len(X_test)))
    for i in range(0, runs):
        model_cpd[i,:] = np.reshape(model.predict(X_test, verbose=0), len(X_test))
    cpd_se = np.std(model_cpd, ddof=1) / np.sqrt(np.size(model_cpd)) # standard error
    c_index_result = concordance_index_censored(e_test, t_test, model_cpd.mean(axis=0))[0] # use mean cpd for c-index
    print(f"Training completed, test loss/C-index/CPD SE: " \
          + f"{round(test_loss, 4)}/{round(c_index_result, 4)}/{round(cpd_se, 4)}")

    # Make predictions by sampling from the Gaussian posterior
    x_pred = X_test[:3]
    for rep in range(5): #Predictions for 5 runs
        print(model.predict(x_pred, verbose=0)[0:3].T)

    if PLOT_SURV_FUNCS:
        # Obtain survival functions
        sample_train_ds = tf.data.Dataset.from_tensor_slices(X_train[..., np.newaxis]).batch(64)
        train_predictions = model.predict(sample_train_ds, verbose=0).reshape(-1)
        breslow = BreslowEstimator().fit(train_predictions, e_train, t_train)

        test_sample = train_test_split(X_test, y_test, e_test, t_test,
                                       test_size=10, stratify=e_test, random_state=0)
        x_sample, y_sample, event_sample, time_sample = test_sample[1::2]
        sample_test_ds = tf.data.Dataset.from_tensor_slices(x_sample[..., np.newaxis]).batch(64)
        test_predictions = model.predict(sample_test_ds, verbose=0).reshape(-1)
        test_surv_fn = breslow.get_survival_function(test_predictions)

        styles = ('-', '--')
        plt.figure(figsize=(6, 4.5))
        for surv_fn, lbl in zip(test_surv_fn, event_sample):
            plt.step(surv_fn.x, surv_fn.y, where="post",
                    linestyle=styles[int(lbl)])
        plt.ylim(0, 1)
        plt.ylabel("Probability of survival $P(T > t)$")
        plt.xlabel("Time $t$")
        plt.legend()
        plt.grid()
        plt.show()