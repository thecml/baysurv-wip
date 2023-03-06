import tensorflow as tf
from tools.model_builder import make_baseline_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from tools import data_loader, model_trainer
from utility.config import load_config
import os
from pathlib import Path
import paths as pt

if __name__ == "__main__":
    # Load config
    config = load_config(pt.CONFIGS_DIR, "baseline.yaml")
    train_size = config['train_size']
    optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
    custom_objects = {"CoxPHLoss": CoxPHLoss()}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loss_fn = tf.keras.losses.deserialize(config['loss_fn'])
    activation_fn = config['activiation_fn']
    num_epochs = config['num_epochs']
    layers = config['network_layers']
    batch_size = config['batch_size']
    dropout_rate = config['dropout_rate']
    regularization_pen = config['l2_kernel_regularization']

    # Load data
    dl = data_loader.GbsgDataLoader().load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = dl.prepare_data(train_size=train_size)
    t_train, t_valid, t_test, e_train, e_valid, e_test = dl.make_time_event_split(y_train, y_valid, y_test)

    model = make_baseline_model(input_shape=X_train.shape[1:],
                                output_dim=1,
                                layers=layers,
                                activation_fn=activation_fn,
                                dropout_rate=dropout_rate,
                                regularization_pen=regularization_pen)

    train_fn = InputFunction(X_train, t_train, e_train, batch_size=batch_size,
                             drop_last=True, shuffle=True)
    valid_fn = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)
    test_fn = InputFunction(X_test, t_test, e_test, batch_size=batch_size)
    
    train_ds = train_fn()
    valid_ds = valid_fn()
    test_ds = test_fn()
    
    trainer = model_trainer.Trainer(model=model,
                                    train_dataset=train_ds,
                                    valid_dataset=valid_ds,
                                    test_dataset=test_ds,
                                    optimizer=optimizer,
                                    loss_function=loss_fn,
                                    num_epochs=num_epochs)
    trainer.train_and_evaluate()
    
    # Save model weights
    model = trainer.model
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    model.save_weights(f'{root_dir}/models/baseline/')
