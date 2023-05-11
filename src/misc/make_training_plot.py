import pandas as pd
import paths as pt
from pathlib import Path
from utility import plot

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"baysurv_results.csv")
    results = pd.read_csv(path)
    results = results.round(3)    
    plot.plot_training_curves(results, 2, "SEER")
    