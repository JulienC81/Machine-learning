# -*- coding=utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import AdaBoostRegressor

## Importation des fichiers qui nous interessent

names = ['API','Surf_X','Surf_Y','Date_Drilling','Date_Completion','Date_Production','Lateral_Length','Depth_TVD_PPLS','Erosion_PPLS','Pressure_PPLS','TOC_PPLS','Vcarb_PPLS','Vsand_PPLS','Vclay_PPLS','PR_PPLS','YM_PPLS','RHOB_PPLS','Res_PPLS','GR_PPLS','DT_PPLS','DTs_PPLS','Temperature','Temp_Anomaly','S3Tect_PPLS','S3_contrast_PPLS','Heat_Flow','Zone','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing','GasCum360','OilCum360']

df_data = pd.read_csv('./TrainSample.csv', 
                        header = None, 
                        sep = ';',
                        decimal = ',',
                        names = names,
                        skiprows = 1,
                        index_col=0,
                        parse_dates = ['Date_Drilling','Date_Completion','Date_Production'],
                        dtype = {col: np.float32 for col in names}
                       )

df_test = pd.read_csv('./TestSample.csv',
                      header = None, 
                      sep = ';', 
                      decimal = ',', 
                      names = names,
                      skiprows = 1,
                      index_col=0,
                      parse_dates = ['Date_Drilling','Date_Completion','Date_Production'],
                      dtype = {col: np.float32 for col in names}
                     )

## Outliers

df_data.loc[746, 'Frac_Gradient'] = np.nan
df_data.loc[553, 'ISIP'] = np.nan
df_data.loc[681, 'Min_Rate_Slurry'] = np.nan
df_data.loc[424, 'Res_PPLS'] = np.nan
df_data.loc[456, 'ShutInPressure_Fil'] = np.nan
df_data.loc[723, 'Min_Treating_Pressure'] = np.nan
df_data.loc[205, 'Min_Treating_Pressure'] = np.nan

##Definition des labels

y_gas = {'GasCum360' : df_data['GasCum360']}
y_oil = {'OilCum360' : df_data['OilCum360']}

df_target_gas = pd.DataFrame(y_gas)
df_target_oil = pd.DataFrame(y_oil)

## Suppression des composantes pas utilisees

cols_date = ['Date_Drilling','Date_Completion','Date_Production', 'GasCum360', 'OilCum360']

df_data.drop(cols_date, 1, inplace=True)
df_test.drop(cols_date, 1, inplace=True)

## Imputation des valeurs manquantes

df_data.dropna()
df_test.dropna()
df_data = df_data.fillna(df_data.mean())
df_test = df_test.fillna(df_data.mean())

from sklearn import cross_validation

X_train, X_test, y_train_gas, y_test_gas = cross_validation.train_test_split(df_data, df_target_gas, test_size=0.2, random_state=0)
X_train, X_test, y_train_oil, y_test_oil = cross_validation.train_test_split(df_data, df_target_oil, test_size=0.2, random_state=0)

## Algorithme - Choix des paramètres 

from sklearn.ensemble import GradientBoostingRegressor

regr_gas = GradientBoostingRegressor()
regr_oil = GradientBoostingRegressor()

regr_gas.fit(X_train, y_train_oil.values)

from matplotlib import pyplot

indices = np.argsort(regr_gas.feature_importances_)[::-1][:41]
g = sns.barplot(y=X_train.columns[indices][:41],x = regr_gas.feature_importances_[indices][:41] , orient='h')