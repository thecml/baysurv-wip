import tensorflow as tf
import tensorflow_probability as tfp
import official.nlp.modeling.layers as nlp_layers

class MonteCarloDropout(tf.keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)

tfd = tfp.distributions
tfb = tfp.bijectors
kl = tfd.kullback_leibler

NUM_MC_SAMPLES = 10

@kl.RegisterKL(tfd.Normal, tfd.HalfCauchy)
def _mc_kl_msf_msf(a, b, seed=None, name='_mc_kl_norm_hc'):
  with tf.name_scope(name):
    s = a.sample(NUM_MC_SAMPLES, seed)
    return tf.reduce_mean(
        a.log_prob(s) - b.log_prob(s), axis=0, name='KL_NORM_HC')

def normal_loc(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def normal_loc_scale(params):
    return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))

def normal_fs(params):
    return tfd.Normal(loc=params[:,0:1], scale=1)

def get_horseshoe_prior(dtype, shape, name, trainable, add_variable_fn):
  horseshoe_prior = tfp.distributions.Independent(tfp.distributions.HalfCauchy(
                                      loc = tf.zeros(shape, dtype = dtype),
                                      scale = 1/tf.sqrt(1.0 * tf.ones(shape, dtype = dtype))),
                                      reinterpreted_batch_ndims = 1)
  return horseshoe_prior

def make_mlp_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    if output_dim == 2: # If 2, then model aleatoric uncertain.
        params = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        dist = tfp.layers.DistributionLambda(normal_loc_scale)(params)
        model = tf.keras.Model(inputs=inputs, outputs=dist)
    else: # Do not model aleatoric uncertain
        output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_vi_model(n_train_samples, input_shape, output_dim, layers, activation_fn, dropout_rate):
    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    bias_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_train_samples * 1.0)
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            bias_divergence_fn=bias_divergence_fn,activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
                
    if output_dim == 2: # If 2, then model both aleatoric and epistemic uncertain.
        params = tfp.layers.DenseFlipout(output_dim,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn)(hidden)
        dist = tfp.layers.DistributionLambda(normal_loc_scale)(params)
        model = tf.keras.Model(inputs=inputs, outputs=dist)
    else: # model only epistemic uncertain.
        output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_mcd_model(input_shape, output_dim, layers,
                   activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(inputs)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(inputs)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn,
                                               activity_regularizer=tf.keras.regularizers.L2(regularization_pen))(hidden)
            else:
                hidden = tf.keras.layers.Dense(units, activation=activation_fn)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = MonteCarloDropout(dropout_rate)(hidden)
    if output_dim == 2: # If 2, then model both aleatoric and epistemic uncertain.
        params = tf.keras.layers.Dense(output_dim)(hidden)
        dist = tfp.layers.DistributionLambda(normal_loc_scale)(params)
        model = tf.keras.Model(inputs=inputs, outputs=dist)        
    else: # model only epistemic uncertain.
        output = tf.keras.layers.Dense(output_dim, activation="linear")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def make_sngp_model(input_shape, output_dim, layers, activation_fn, dropout_rate, regularization_pen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    spec_norm_bound = 0.9
    for i, units in enumerate(layers):
        if i == 0:
            if regularization_pen is not None:
                dense = tf.keras.layers.Dense(units, activation=activation_fn,
                                              activity_regularizer=tf.keras.regularizers.L2(regularization_pen))
            else:
                dense = tf.keras.layers.Dense(units, activation=activation_fn)
            dense = nlp_layers.SpectralNormalization(dense, norm_multiplier=spec_norm_bound)
            hidden = dense(inputs)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
        else:
            if regularization_pen is not None:
                dense = tf.keras.layers.Dense(units, activation=activation_fn,
                                              activity_regularizer=tf.keras.regularizers.L2(regularization_pen))
            else:
                dense = tf.keras.layers.Dense(units, activation=activation_fn)
            dense = nlp_layers.SpectralNormalization(dense, norm_multiplier=spec_norm_bound)
            hidden = dense(hidden)
            if dropout_rate is not None:
                hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
            
    output = nlp_layers.RandomFeatureGaussianProcess(units=output_dim,
                                                     num_inducing=1024,
                                                     normalize_input=False,
                                                     scale_random_features=True,
                                                     gp_cov_momentum=-1)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model
    
    
    
        
                        