def get_baseline_sweep_config():
    return {
        "method": "bayes",
        "metric": {
        "name": "val_loss",
        "goal": "minimize"
        },
        "parameters": {
            "network_layers": {
                "values": [[8], [8, 8], [8, 8, 8],
                           [16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64]]
            },
            "learning_rate": {
                "values": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
            },
            "optimizer": {
                "values": ["Adam", "RMSprop", "SGD", "Nadam"],
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [None, 0.1, 0.25, 0.5, 0.6]
            }
        }
    }