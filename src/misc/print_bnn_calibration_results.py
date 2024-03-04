import pandas as pd
import paths as pt
from pathlib import Path
import glob
import os
from utility.model import map_model_name

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"baysurv_test_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(2)
    
    model_names = ["mlp", "sngp", "vi","mcd1", "mcd2", "mcd3"]
    dataset_names = ["MIMIC"]
    
    for dataset_name in dataset_names:
        for index, model_name in enumerate(model_names):
            if index > 0:
                text = "+ "
            else:
                text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            ici = float(res['ICI'])
            d_calib = float(res['DCalib'])
            c_calib = float(res['CCalib'])
            km = float(res['KM'])
            if model_name in ["MLP"]:
                c_calib = "-"
            model_name = map_model_name(model_name)
            text += f"{model_name} & "
            text += f"{ici} & {d_calib} & {c_calib} & {km} \\\\"
            print(text)
        print()