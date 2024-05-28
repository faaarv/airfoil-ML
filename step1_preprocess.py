import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler

import joblib

df = pd.read_csv('dataset.csv')
scaler = MinMaxScaler()
# scaler = StandardScaler()

columns_to_normalize = ['Cd',  'Cl'  , 'Ncrit' , 'Re' , 'alpha']  
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df_normalized = pd.DataFrame(df, columns=df.columns)
df_normalized.to_csv('dataset_normalized.csv', index=False)

scaler_train_filename = 'scaler_train_MM.pkl'
# scaler_train_filename = 'scaler_train_std.pkl'

joblib.dump(scaler, scaler_train_filename)
