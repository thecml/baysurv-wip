import pandas as pd
import paths as pt
from pathlib import Path

def map_model_name(model_name):
    if model_name == "cox":
        model_name = "CoxPH"
    if model_name == "coxnet":
        model_name = "CoxNet"
    if model_name == "coxboost":
        model_name = "CoxBoost"
    if model_name == "rsf":
        model_name = "Random Survival Forest"
    if model_name == "dcm":
        model_name = "Deep Cox Mixtures"
    if model_name == "dsm":
        model_name = "Deep Survival Machines"
    if model_name == "baycox":
        model_name = "BayesianCox"
    if model_name == "baymtlr":
        model_name = "BayesianMTLR"
    return model_name

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(3)

    model_names = results['ModelName'].unique()
    dataset_names = ["SEER"] #"SUPPORT", "SEER", "METABRIC"

    for dataset_name in dataset_names:
        for index, model_name in enumerate(model_names):
            text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            loss = float(res['Loss'])
            ci = float(res['CI'])
            ibs = float(res['IBS'])
            mae = float(res['MAE'])
            d_calib = float(res['DCalib'])
            km = float(res['KM'])
            inbll = float(res['INBLL'])
            c_calib = float(res['CCalib'])
            ici = float(res['ICI'])
            e50 = float(res['E50'])
            if loss != loss:
                loss = "NA"
            model_name = map_model_name(model_name)
            text += f"{model_name} & {mae} & {ci} & {ibs} & {inbll} & {loss} & {ici} & {e50} & {d_calib} & {c_calib} & {km} \\\\"
            print(text)
        print()
