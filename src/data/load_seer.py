import pandas as pd
import numpy as np
import paths as pt
from pathlib import Path
from utility.survival import convert_to_structured
from tools.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from tools.model_builder import make_coxnet_model
from utility.metrics import concordance_index_censored

if __name__ == "__main__":
    path = Path.joinpath(pt.DATA_DIR, "seer.csv")
    df = pd.read_csv(path, na_values="Blank(s)")
    df = df.loc[:, df.isin([' ','NULL',0]).mean() < .2] # drop columns with more than 20% nan
    
    df['SEER cause-specific death classification'] = df['SEER cause-specific death classification'] \
                                                    .apply(lambda x: 1 if x=="Dead (attributable to this cancer dx)" else 0)
    df = df.loc[df['Survival months'] != "Unknown"].copy(deep=True)
    df['Survival months'] = df['Survival months'].astype(int)
    
    X = df.drop(['Survival months', 'SEER cause-specific death classification'], axis=1)
    y = convert_to_structured(df['Survival months'], df['SEER cause-specific death classification'])

    X = X.dropna(axis=1) # remove nan only cols

    obj_cols = df.select_dtypes(['bool']).columns.tolist() \
                + df.select_dtypes(['object']).columns.tolist()
    for col in obj_cols:
        X[col] = X[col].astype('category')
    
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(['category']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train) # 83378 x 549
    X_test = transformer.transform(X_test)
    
    model = make_coxnet_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    c_index = concordance_index_censored(y_test["Event"], y_test["Time"], predictions)[0]   
    print(c_index)


        
    