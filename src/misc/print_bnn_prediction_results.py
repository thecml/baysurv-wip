import pandas as pd
import paths as pt
from pathlib import Path
import glob
import os

def map_model_name(model_name):
    if model_name == "MLP":
        model_name = "Baseline (MLP)"
    elif model_name == "MLP-ALEA":
        model_name = "Aleatoric"
    elif model_name == "MCD-EPI":
        model_name = "Epistemic"
    else:
        model_name = "Both"
    return model_name

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"baysurv_test_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(3)
    
    model_names = ["MLP", "MLP-ALEA", "MCD-EPI", "MCD"]
    dataset_names = ["METABRIC", "SEER", "SUPPORT", "MIMIC"]
    
    for dataset_name in dataset_names:
        for index, model_name in enumerate(model_names):
            if index > 0:
                text = "+ "
            else:
                text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            ci = float(res['CI'])
            mae_h = float(res['MAEHinge'])
            mae_po = float(res['MAEPseudo'])
            ibs = float(res['IBS'])
            model_name = map_model_name(model_name)
            text += f"{model_name} & "
            text += f"{ci} & {mae_h} & {mae_po} & {ibs} \\\\"
            print(text)
        print()
        