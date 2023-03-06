def get_baseline_sweep_config():
    return {
        "method": "random", # try grid or random
        "metric": {
        "name": "val_loss",
        "goal": "minimize"
        },
        "parameters": {
            "batch_size": {
                "values": [8, 16, 32, 64, 128]
            },
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64],
                           [64, 32], [64, 32, 32], [64, 16, 16],
                           [32, 16], [32, 16, 16], [16, 16, 16]]
            },
            "learning_rate": {
                "values": [0.0001, 0.001, 0.01, 0.1],
            },
            "optimizer": {
                "values": ["Adam", "RMSprop", "SGD", "Nadam"],
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [None, 0.25, 0.5]
            },
            "l2_kernel_regularization": {
                "values": [None, 0.1, 0.01, 0.001]
            }
        }
    }

def get_vi_sweep_config():
    return {
        "method": "random", # try grid or random
        "metric": {
        "name": "val_loss",
        "goal": "minimize"
        },
        "parameters": {
            "batch_size": {
                "values": [8, 16, 32, 64, 128]
            },
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64],
                           [64, 32], [64, 32, 32], [64, 16, 16],
                           [32, 16], [32, 16, 16], [16, 16, 16]]
            },
            "learning_rate": {
                "values": [0.0001, 0.001, 0.01, 0.1],
            },
            "optimizer": {
                "values": ["Adam", "RMSprop", "SGD", "Nadam"],
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [None, 0.25, 0.5]
            },
        }
    }

def get_mc_dropout_sweep_config():
    return {
        "method": "random", # try grid or random
        "metric": {
        "name": "val_loss",
        "goal": "minimize"
        },
        "parameters": {
            "batch_size": {
                "values": [8, 16, 32, 64, 128]
            },
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64],
                           [64, 32], [64, 32, 32], [64, 16, 16],
                           [32, 16], [32, 16, 16], [16, 16, 16]]
            },
            "learning_rate": {
                "values": [0.0001, 0.001, 0.01, 0.1],
            },
            "optimizer": {
                "values": ["Adam", "RMSprop", "SGD", "Nadam"],
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [0.25, 0.5]
            },
            "l2_kernel_regularization": {
                "values": [None, 0.1, 0.01, 0.001]
            }
        }
    }