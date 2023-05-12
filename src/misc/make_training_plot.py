import pandas as pd
import paths as pt
import glob
import os
from utility import plot

if __name__ == "__main__":
    path = pt.RESULTS_DIR
    all_files = glob.glob(os.path.join(path , "baysurv*.csv"))
    
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    results = pd.concat(li, axis=0, ignore_index=True)
    
    results = results.round(3)
    for dataset_name in  ["WHAS500", "SEER", "GBSG2", "FLCHAIN", "SUPPORT", "METABRIC"]:
        plot.plot_training_curves(results, 10, dataset_name)
    