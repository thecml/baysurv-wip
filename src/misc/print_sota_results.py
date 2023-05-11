import pandas as pd
import paths as pt
from pathlib import Path

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv")
    results = pd.read_csv(path)
    
    results = results.round(3)
    
    model_names = results['ModelName'].unique()
    dataset_names_list = [["WHAS500", "SEER", "GBSG2"],  ["FLCHAIN", "SUPPORT", "METABRIC"]]
    
    for dataset_names in dataset_names_list:
        for model_name in model_names:
            text = ""
            for index, ds in enumerate(dataset_names):
                t_train = float(results.loc[(results['DatasetName'] == ds) & (results['ModelName'] == model_name)]['TrainTime'])
                ci = float(results.loc[(results['DatasetName'] == ds) & (results['ModelName'] == model_name)]['TestCI'])
                ctd = float(results.loc[(results['DatasetName'] == ds) & (results['ModelName'] == model_name)]['TestCTD'])
                ibs = float(results.loc[(results['DatasetName'] == ds) & (results['ModelName'] == model_name)]['TestIBS'])
                loss = float(results.loc[(results['DatasetName'] == ds) & (results['ModelName'] == model_name)]['TestLoss'])
                if loss != loss:
                    loss = "NA"
                if index == 0:
                    text += f"{model_name} & "
                if index == 2:
                    text += f"{t_train} & {ci} & {ctd} & {ibs} & {loss} \\\\"
                else:
                    text += f"{t_train} & {ci} & {ctd} & {ibs} & {loss} & "
            print(text)
        
    