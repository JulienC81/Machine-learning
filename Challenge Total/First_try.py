
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


names = ['API',
         'Surf_X',
         'Surf_Y',
         'Date_Drilling',
         'Date_Completion', 
         'Date_Production', 
         'Lateral_Length', 
         'Depth_TVD_PPLS', 
         'Erosion_PPLS', 
         'Pressure_PPLS', 
         'TOC_PPLS', 
         'Vcarb_PPLS', 
         'Vsand_PPLS', 
         'Vclay_PPLS', 
         'PR_PPLS', 
         'YM_PPLS', 
         'RHOB_PPLS', 
         'Res_PPLS', 
         'GR_PPLS', 
         'DT_PPLS', 
         'DTs_PPLS', 
         'Temperature', 
         'Temp_Anomaly', 
         'S3Tect_PPLS', 
         'S3_contrast_PPLS', 
         'Heat_Flow', 
         'Zone', 
         'Nbr_Stages', 
         'Frac_Gradient', 
         'Proppant_Designed', 
         'Proppant_in_Formation', 
         'Avg_Breakdown_Pressure', 
         'Avg_Treating_Pressure', 
         'Max_Treating_pressure', 
         'Min_Treating_Pressure', 
         'Avg_Rate_Slurry', 
         'Max_Rate_Slurry', 
         'Min_Rate_Slurry', 
         'ShutInPressure_Fil', 
         'ShutInPressure_Initial', 
         'ISIP', 
         'Shot_Density', 
         'Shot_Total', 
         'Proppant_per_ft', 
         'Stage_Spacing', 
         'GasCum360', 
         'OilCum360']

df_data = pd.read_csv('./TrainSample.csv',
                        header = None,
                        sep = ';',
                        decimal = ',',
                        names = names,
                        skiprows = 1,
                        na_filter = True,
                        parse_dates = ['Date_Drilling','Date_Completion','Date_Production'],
                        dtype = {col: np.float32 for col in names}
                       )

df_test = pd.read_csv('./TestSample.csv',
                      header = None, 
                      sep = ';', 
                      decimal = ',', 
                      names = names,
                      skiprows = 1,
                      na_filter = True,
                      parse_dates = ['Date_Drilling','Date_Completion','Date_Production'],
                      dtype = {col: np.float32 for col in names}
                     )

#df_data.set_index('API')

df_test.head(5)

##Définition des labels

y = {'GasCum360' : df_data['GasCum360'],
     'OilCum360' : df_data['OilCum360']}

df_target = pd.DataFrame(y)

cols_date = ['Date_Drilling','Date_Completion','Date_Production', 'GasCum360', 'OilCum360']

df_data.drop(cols_date, 1, inplace=True)
df_test.drop(cols_date, 1, inplace=True)

cols_with_nan = ['Pressure_PPLS','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing']

df_data.drop(cols_with_nan, axis=1, inplace=True)
df_test.drop(cols_with_nan, axis=1, inplace=True)

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_data, df_target, test_size=0.2, random_state=0)

#x_testFixed = x_test.fillna(x_test.median())
#x_trainFixed = x_train.fillna(x_train.median())

def correlation(data):
    correlation = data.corr()
    figure = plt.subplots( figsize =( 12 , 10 ) )
    color_map = sns.diverging_palette( 220 , 10 , as_cmap = True )
    figure = sns.heatmap(
        correlation, 
        cmap = color_map, 
        annot = True, 
        annot_kws = { 'fontsize' : 2 }
    )
    
correlation(df_train)

## Fit regression model

regr_1 = KNeighborsRegressor()
regr_1.fit(X_train, y_train)

## Predict

y_predictDec = regr_1.predict(X_test)

## Metrics training

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_predictDec) 
#mean_squared_error(y_test, y_predictAda)

gascumpred = []
oilcumpred = []

gascumpred = [element[0] for element in y_predictFin]
oilcumpred = [element[1] for element in y_predictFin]

gascumpred_max = max(gascumpred)
gascumpred_min = min(gascumpred)

oilcumpred_max = max(oilcumpred)
oilcumpred_min = min(oilcumpred)

GasCum360_inf = gascumpred - (gascumpred_max - gascumpred_min) / 4
GasCum360_sup = gascumpred + (gascumpred_max - gascumpred_min) / 4

OilCum360_inf = oilcumpred - (oilcumpred_max - oilcumpred_min) / 4
OilCum360_sup = oilcumpred + (oilcumpred_max - oilcumpred_min) / 4

print(GasCum360_inf[:10], GasCum360_sup[:10])

GasCum360_inf.tolist
GasCum360_sup.tolist

OilCum360_inf.tolist
OilCum360_sup.tolist

## Output

"""
A FAIRE

Faire une cellule avec le classifier qui prend en entrée toutes les données
du dataset d'entrainement.
Faire la prédiction sur le dataset de test
Mettre en forme les données ainsi prédite pour les exporter dans un CSV

A FAIRE 
"""

id_test = df_test['API'].values.tolist()

output = pd.DataFrame({'API': id_test,
                       'GasCum360_INF': GasCum360_inf,
                       'GasCum360_SUP': GasCum360_sup,
                       'OilCum360_INF': OilCum360_inf,
                       'OilCum360_SUP': OilCum360_sup},
                      index=id_test
                     )

output.head()

output.to_csv('coche-julien-challenge-total.csv', index=False, sep= ';', decimal=',')

"""
Rappport 1 :

Importation correcte des données
Début du travail exploratoire, affichage du head et description des données
Beaucoup de colonnes sont incomplètes => identification et élimination de ces colonnes
Observation des composantes corrélées.

Rapport 2 :

Elimination des colonnes incomplètes
Exploration des données de sorties
Choix de l'algorithme - DecisienTreeRegressor / AdaBoost

Rapport 3 :

Fin elimination des colonnes incompletes
Fin du formatage des données
Debut mise en place de la cross_validation
Debut mise en place des metrics
"""