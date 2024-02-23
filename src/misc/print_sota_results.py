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
    dataset_names = ["SUPPORT", "SEER", "METABRIC", "MIMIC"]
    model_citations = ['\cite{cox_regression_1972}', '\cite{simon_regularization_2011}',
                       '\cite{hothorn_survival_2005}', '\cite{ishwaran_random_2008}',
                       '\cite{nagpal_deep_2021}', '\cite{qi_using_2023}', '\cite{qi_using_2023}']

    for dataset_name in dataset_names:
        for index, (model_name, model_citation) in enumerate(zip(model_names, model_citations)):
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
            if model_name in ["cox", "coxnet", "coxboost", "rsf", "dsm"]:
                c_calib = "-"
            else:
                if c_calib == 1.0:
                    c_calib = "Yes"
                else:
                    c_calib = "No"
            model_name = map_model_name(model_name)
            text += f"{model_name} {model_citation} & "
            text += f"{ci} & {mae_hinge} & {mae_pseudo} & {ibs} & {inbll} & {ici} & {d_calib} & {c_calib} & {km} \\\\"
            print(text)
        print()
