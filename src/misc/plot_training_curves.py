import pandas as pd
import paths as pt
import glob
import os
from utility import plot

DATASETS = ['WHAS500']
MODEL_NAMES = ["MLP", "MLP-ALEA", "VI-EPI", "MCD"]
METRIC_NAMES = ['TrainLoss', 'TrainCTD', 'TrainIBS', 'TrainINBLL']

if __name__ == "__main__":
    path = pt.RESULTS_DIR
    all_files = glob.glob(os.path.join(path , "baysurv_whas500_results.csv"))
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    results = pd.concat(li, axis=0, ignore_index=True)
    results = results.round(3)
    n_epochs = 10
    n_metrics = len(METRIC_NAMES)
    for dataset in DATASETS:
        plot.plot_training_curves(results, n_epochs, dataset, MODEL_NAMES, METRIC_NAMES)
