import numpy as np
import pandas as pd
import os
import numpy as np
from pathlib import Path
import paths as pt
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

def plot_training_curves(results):
    datasets = results['DatasetName'].unique()
    n_epochs = int(results.index.max() + 1)
    for dataset_name in datasets:
        baseline_results = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == "Baseline")]
        mc_results = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == "MC")]
             
        baseline_loss = baseline_results[['Loss']]
        baseline_ci = baseline_results[['CI']]
        baseline_ctd = baseline_results[['CTD']]
        baseline_ibs = baseline_results[['IBS']]
        
        mc_loss = mc_results[['Loss']]
        mc_ci = mc_results[['CI']]
        mc_ctd = mc_results[['CTD']]
        mc_ibs = mc_results[['IBS']]
        
        epochs = range(1, n_epochs+1)
        fig, axs = plt.subplots(1, 4, figsize=(18, 3))
        axs[0].plot(epochs, baseline_loss, label='Baseline')
        axs[0].plot(epochs, mc_loss, label='MC')
        axs[0].legend(loc="upper right")
        axs[0].set(xlabel='Epoch', ylabel=r'Model loss $\mathcal{L}(\theta)$')
        axs[1].plot(epochs, baseline_ci, label='MLP')
        axs[1].plot(epochs, mc_ci, label='MC')
        axs[1].set(xlabel='Epoch', ylabel='C-Index')
        axs[2].plot(epochs, baseline_ctd, label='MLP')
        axs[2].plot(epochs, mc_ctd, label='MC')
        axs[2].set(xlabel='Epoch', ylabel='$C_{td}$')
        axs[3].plot(epochs, baseline_ibs, label='MLP')
        axs[3].plot(epochs, mc_ibs, label='MC')
        axs[3].set(xlabel='Epoch', ylabel='IBS')
        fig.savefig(Path.joinpath(pt.RESULTS_DIR, f"{dataset_name}_training_curves.pdf"),
                    format='pdf', bbox_inches="tight")
    