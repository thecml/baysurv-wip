import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utility.risk import InputFunction
from utility.metrics import CindexMetric
from utility.loss import CoxPHLoss
import tensorflow_probability as tfp

from tools.model_builder import make_vi_model, make_mc_model
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
from tools import data_loader, model_trainer
from sklearn.model_selection import train_test_split
from tools.preprocessor import Preprocessor
from utility.config import load_config
import paths as pt
from pathlib import Path

tfd = tfp.distributions
tfb = tfp.bijectors

N_EPOCHS = 1
BATCH_SIZE = 32

if __name__ == "__main__":
    # Load data
    dl = data_loader.FlchainDataLoader().load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    
    # Scale data
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                one_hot=True, fill_value=-1)
    X_train = np.array(transformer.transform(X_train))
    X_test = np.array(transformer.transform(X_test))
    
    # Make time/event split
    t_train = np.array(y_train['Time'])
    e_train = np.array(y_train['Event'])
    t_test = np.array(y_test['Time'])
    e_test = np.array(y_test['Event'])
    
    # Load network parameters
    config = load_config(pt.CONFIGS_DIR, "gbsg_arch.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    custom_objects = {"CoxPHLoss": CoxPHLoss()}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loss_fn = tf.keras.losses.deserialize(config['loss_fn'])
    layers = config['network_layers']
    activation_fn = config['activiation_fn']
    dropout_rate = config['dropout_rate']
    l2_reg = config['dropout_rate']
    
    # Make data loaders
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)()
    test_ds = InputFunction(X_test, t_test, e_test, batch_size=BATCH_SIZE)()
    
    # Train VI model
    model = make_mc_model(input_shape=X_train.shape[1:], output_dim=2,
                          layers=layers, activation_fn=activation_fn,
                          dropout_rate=dropout_rate)
    trainer = model_trainer.Trainer(model=model,
                                    model_type="MC",
                                    train_dataset=train_ds,
                                    valid_dataset=None,
                                    test_dataset=test_ds,
                                    optimizer=optimizer,
                                    loss_function=loss_fn,
                                    num_epochs=N_EPOCHS)
    trainer.train_and_evaluate()
    
    fig = plt.figure()
    epochs = range(1, N_EPOCHS+1)
    plt.plot(epochs, trainer.test_loss_scores, label='MC Loss')
    fig.savefig(Path.joinpath(pt.RESULTS_DIR, "loss.pdf"),
                format='pdf', bbox_inches="tight")

    '''
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
    '''