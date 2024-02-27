import pandas as pd
import paths as pt
from pathlib import Path
import glob
import os

def map_model_name(model_name):
    if model_name == "cox":
        model_name = "CoxPH"
    if model_name == "coxnet":
        model_name = "CoxNet"
    if model_name == "coxboost":
        model_name = "CoxBoost"
    if model_name == "rsf":
        model_name = "RSF"
    if model_name == "dsm":
        model_name = "DSM"
    if model_name == "dcm":
        model_name = "DCM"
    if model_name == "baycox":
        model_name = "BayCox"
    if model_name == "baymtlr":
        model_name = "BayMTLR"
    return model_name

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(2)
    
    model_names = ["cox", "coxnet", "coxboost", "rsf", "dsm", "dcm", "baycox", "baymtlr"]
    dataset_names = ["METABRIC", "SEER", "SUPPORT", "MIMIC"]
    model_citations = ['\cite{cox_regression_1972}', '\cite{simon_regularization_2011}',
                       '\cite{hothorn_survival_2005}', '\cite{ishwaran_random_2008}',
                       '\cite{nagpal_deep_2021}', '\cite{nagpal_deep_cox_2021}',
                       '\cite{qi_using_2023}', '\cite{qi_using_2023}']
    
    for dataset_name in dataset_names:
        for index, (model_citation ,model_name) in enumerate(zip(model_citations, model_names)):
            text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            if res.empty:
                break
            ci = float(res['CI'])
            mae_h = float(res['MAEHinge'])
            mae_po = float(res['MAEPseudo'])
            ibs = float(res['IBS'])
            model_name = map_model_name(model_name)
            text += f"{model_name} {model_citation} & "
            text += f"{ci} & {mae_h} & {mae_po} & {ibs} \\\\"
            print(text)
        print()
        