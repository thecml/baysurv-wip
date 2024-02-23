import pandas as pd
import paths as pt
from pathlib import Path
import glob
import os

def map_model_name(model_name):
    if model_name == "MLP":
        model_name = "Baseline (MLP)"
    return model_name

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"baysurv_test_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(3)
    
    model_names = ["MLP", "VI", "MCD", "SNGP"]
    dataset_names = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
    
    for dataset_name in dataset_names:
        for index, model_name in enumerate(model_names):
            if index > 0:
                text = "+ "
            else:
                text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            ci = float(res['CI'])
            ibs = float(res['IBS'])
            mae_hinge = float(res['MAEHinge'])
            mae_pseudo = float(res['MAEPseudo'])
            km = float(res['KM'])
            inbll = float(res['INBLL'])
            d_calib = float(res['DCalib'])
            c_calib = float(res['CCalib'])
            ici = float(res['ICI'])
            if d_calib == 1.0:
                d_calib = "Yes"
            else:
                d_calib = "No"
            if model_name in ["MLP", "SNGP"]:
                c_calib = "-"
            else:
                if c_calib == 1.0:
                    c_calib = "Yes"
                else:
                    c_calib = "No"
            model_name = map_model_name(model_name)
            text += f"{model_name} & "
            text += f"{ci} & {mae_hinge} & {mae_pseudo} & {ibs} & {inbll} & {ici} & {d_calib} & {c_calib} & {km} \\\\"
            print(text)
        print()
        