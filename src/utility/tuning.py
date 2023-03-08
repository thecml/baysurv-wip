def get_baseline_sweep_config():
    return {
        "method": "bayes",
        "metric": {
        "name": "val_ci",
        "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64]]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1, 1],
            },
            "optimizer": {
                "values": ["Adam", "Nadam", "Adagrad", "Adadelta", "Adamax"],
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [None, 0.25, 0.5]
            },
            "l2_reg": {
                "values": [None, 0.001, 0.01, 0.1]
            }
        }
    }