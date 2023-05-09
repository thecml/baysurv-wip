import tensorflow as tf
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from utility.config import load_config
import paths as pt
import numpy as np
import random
from utility.training import get_data_loader, scale_data, make_time_event_split
from sklearn.model_selection import train_test_split
from tools.model_builder import make_mlp_model, make_mcd_model
from tools.model_trainer import Trainer

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

N_EPOCHS = 25
MODEL_TYPE = "MLP"
DATASET = "METABRIC"

if __name__ == "__main__":
    # Load config
    config = load_config(pt.MLP_CONFIGS_DIR, f"{DATASET.lower()}.yaml")
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    custom_objects = {"CoxPHLoss": CoxPHLoss()}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loss_fn = tf.keras.losses.deserialize(config['loss_fn'])
    activation_fn = config['activiation_fn']
    layers = config['network_layers']
    dropout_rate = config['dropout_rate']
    l2_reg = config['l2_reg']
    batch_size = config['batch_size']

    # Load data
    dl = get_data_loader(DATASET).load_data()
    X, y = dl.get_data()
    num_features, cat_features = dl.get_features()

    # Split data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # Scale data
    X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Make time/event split
    t_train, e_train = make_time_event_split(y_train)
    t_test, e_test = make_time_event_split(y_test)

    # Make event times
    lower, upper = np.percentile(t_test[t_test.dtype.names], [10, 90])
    event_times = np.arange(lower, upper+1)

    # Make data loaders
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size, drop_last=True, shuffle=True)()
    test_ds = InputFunction(X_test, t_test, e_test, batch_size=batch_size)()

    mlp_model = make_mcd_model(input_shape=X_train.shape[1:], output_dim=2,
                               layers=layers, activation_fn=activation_fn,
                               dropout_rate=dropout_rate, regularization_pen=l2_reg)
    trainer = Trainer(model=mlp_model, model_type=MODEL_TYPE,
                      train_dataset=train_ds, valid_dataset=None,
                      test_dataset=test_ds, optimizer=optimizer,
                      loss_function=loss_fn, num_epochs=N_EPOCHS,
                      event_times=event_times)
    trainer.train_and_evaluate()

    # Test
    test_ci = trainer.test_ci_scores
    test_ctd = trainer.test_ctd_scores
    test_ibs = trainer.test_ibs_scores

    print(test_ci[-1])
    print(test_ctd[-1])
    print(test_ibs[-1])
