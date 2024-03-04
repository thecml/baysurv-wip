import pandas as pd
import paths as pt
import glob
import os
from utility import plot
from pathlib import Path

DATASETS = ['SEER']
MODEL_NAMES = ["mlp", "sngp", "vi", "mcd1", "mcd2", "mcd3"]
METRIC_NAMES = ["TrainLoss", "TrainVariance", "ValidLoss", "ValidVariance"]

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"baysurv_training_results.csv")
    results = pd.read_csv(path)
    results = results.round(3)
    for dataset in DATASETS:
        plot.plot_training_curves(results, dataset, MODEL_NAMES, METRIC_NAMES)
