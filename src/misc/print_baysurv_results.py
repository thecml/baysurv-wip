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
    path = pt.RESULTS_DIR    
    all_files = glob.glob(os.path.join(path , "baysurv*.csv"))
    
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    results = pd.concat(li, axis=0, ignore_index=True)
    results = results.round(3)
    
    model_names = ["MLP", "VI", "MCD"]
    dataset_names = ["SEER"] #"SUPPORT", "SEER", "METABRIC"
    
    for dataset_name in dataset_names:
        for index, model_name in enumerate(model_names):
            if index > 0:
                text = "+ "
            else:
                text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            #best_ep = res.iloc[0]['BestEpoch']
            #best_obs = res.groupby(['ModelName', 'DatasetName'])[metrics].nth(best_ep-1)
            #t_train = res.groupby(['ModelName', 'DatasetName'])['TrainTime'].nth(range(best_ep)).sum()            
            loss = float(res['Loss'])
            ci = float(res['CI'])
            ibs = float(res['IBS'])
            mae = float(res['MAE'])
            d_calib = float(res['DCalib'])
            km = float(res['KM'])
            inbll = float(res['INBLL'])
            c_calib = float(res['Calib'])
            ici = float(res['ICI'])
            e50 = float(res['E50'])
            model_name = map_model_name(model_name)
            text += f"{model_name} & "
            text += f"{model_name} & {mae} & {ci} & {ibs} & {inbll} & {loss} & {ici} & {e50} & {d_calib} & {c_calib} & {km} \\\\"
            print(text)
        print()
        