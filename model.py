from   category_encoders          import *
import numpy as np
import pandas as pd

from   sklearn.compose            import *
from   sklearn.ensemble           import RandomForestClassifier
from   sklearn.metrics            import accuracy_score 
from   sklearn.pipeline           import Pipeline
from   sklearn.preprocessing      import *
from   sklearn.tree               import DecisionTreeClassifier, ExtraTreeClassifier
from   sklearn.model_selection    import train_test_split
# Imputing
from   sklearn.impute             import KNNImputer
# Saving model
import pickle

import warnings
warnings.filterwarnings('ignore')

## EDA

df = pd.read_csv('ufc-master.csv')

## Drop Columns Not Used for Model Prediction

class HandPickedFeatureFilter:
    def __init__(self):
        self.picked_feats = [
            'R_odds', 'B_odds', 'gender',
            'B_avg_SIG_STR_landed', 'R_avg_SIG_STR_landed',
            'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct',
            'B_avg_SUB_ATT', 'R_avg_SUB_ATT',
            'B_avg_TD_landed', 'R_avg_TD_landed',
            'B_avg_TD_pct', 'R_avg_TD_pct',
            'B_losses', 'R_losses',
            'B_wins', 'R_wins',
            'B_Stance', 'R_Stance',
            'B_Height_cms', 'R_Height_cms',
            'B_Reach_cms', 'R_Reach_cms',
            'B_age', 'R_age','Winner']
    
    def transform(self, X, **transform_params):
        df_copy = X.copy()
        df_copy = df_copy.filter(self.picked_feats)
        return df_copy

    def fit(self, X, y=None, **fit_params):
        return self

class EncodeVars:
    def __init__(self):
        pass
    
    def transform(self, X, **transform_params):
        # Encode "Blue" as 0 and "Red" as 1
        def encode_win(x):
            if x is np.nan:
                return x
            if x == 'neither':
                return np.nan
            if x.lower() == 'blue':
                return 0
            elif x.lower() == 'red':
                return 1
            
        # Remove white space
        def encode_stance(x):
            return x.strip()
        
        df_copy = X.copy()
        df_copy['Winner'] = df_copy['Winner'].apply(encode_win)
        df_copy['B_Stance'] = df_copy['B_Stance'].apply(encode_stance)
        return df_copy

    def fit(self, X, y=None, **fit_params):
        return self

pipe = Pipeline([('hand_pick', HandPickedFeatureFilter()),
                 ('encode_vars', EncodeVars())])
X = pipe.transform(df).iloc[:,:-1]
y = pipe.transform(df).iloc[:,-1]

### Create Pipeline

categorical_cols = (X.dtypes == object)

def UFCPipe(clf):
    con_pipe = Pipeline([('imputer', KNNImputer(n_neighbors=14, missing_values=np.nan)), 
                         ('scaler', StandardScaler())])

    cat_pipe = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessing = ColumnTransformer([('categorical', cat_pipe,  categorical_cols), 
                                       ('continuous',  con_pipe, ~categorical_cols)])

    pipe = Pipeline([('preprocessing', preprocessing), 
                     ('clf', clf)])
    return pipe

best_hypers = {'n_estimators': 522, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'criterion': 'gini'}

## Train Final Model with all Data

pipe = UFCPipe(RandomForestClassifier(**best_hypers, n_jobs=-1, random_state=11))
pipe.fit(X, y)
# with open('model.pkl', 'wb') as f:
#     pickle.dump(pipe, f)