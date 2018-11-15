
# coding: utf-8

# In[539]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR

# In[540]:

names = ['API','Surf_X','Surf_Y','Date_Drilling','Date_Completion','Date_Production','Lateral_Length','Depth_TVD_PPLS','Erosion_PPLS','Pressure_PPLS','TOC_PPLS','Vcarb_PPLS','Vsand_PPLS','Vclay_PPLS','PR_PPLS','YM_PPLS','RHOB_PPLS','Res_PPLS','GR_PPLS','DT_PPLS','DTs_PPLS','Temperature','Temp_Anomaly','S3Tect_PPLS','S3_contrast_PPLS','Heat_Flow','Zone','Nbr_Stages','Frac_Gradient','Proppant_Designed','Proppant_in_Formation','Avg_Breakdown_Pressure','Avg_Treating_Pressure','Max_Treating_pressure','Min_Treating_Pressure','Avg_Rate_Slurry','Max_Rate_Slurry','Min_Rate_Slurry','ShutInPressure_Fil','ShutInPressure_Initial','ISIP','Shot_Density','Shot_Total','Proppant_per_ft','Stage_Spacing','GasCum360','OilCum360']

df_data = pd.read_csv('./TrainSample.csv', 
                        header = None, 
                        sep = ';',
                        decimal = ',',
                        names = names,
                        skiprows = 1,
                        parse_dates = ['Date_Drilling','Date_Completion','Date_Production'],
                        dtype = {col: np.float32 for col in names}
                       )

df_test = pd.read_csv('./TestSample.csv',
                      header = None, 
                      sep = ';', 
                      decimal = ',', 
                      names = names,
                      skiprows = 1,
                      parse_dates = ['Date_Drilling','Date_Completion','Date_Production'],
                      dtype = {col: np.float32 for col in names}
                     )

#df_data.set_index('API')

df_test.head(5)


# In[541]:

data_mean_gas = df_data[['GasCum360']].mean()
data_mean_oil = df_data[['OilCum360']].mean()


# In[542]:

##Définition des labels

y_gas = {'GasCum360' : df_data['GasCum360']}
y_oil = {'OilCum360' : df_data['OilCum360']}

df_target_gas = pd.DataFrame(y_gas)
df_target_oil = pd.DataFrame(y_oil)


# In[543]:

cols_date = ['Date_Drilling','Date_Completion','Date_Production', 'GasCum360', 'OilCum360']

df_data.drop(cols_date, 1, inplace=True)
df_test.drop(cols_date, 1, inplace=True)


# In[544]:

df_data.dropna()
df_test.dropna()
df_data = df_data.fillna(df_data.mean())
df_test = df_test.fillna(df_data.mean())


# In[545]:

from sklearn import cross_validation

X_train, X_test, y_train_gas, y_test_gas = cross_validation.train_test_split(df_data, df_target_gas, test_size=0.2, random_state=0)
X_train, X_test, y_train_oil, y_test_oil = cross_validation.train_test_split(df_data, df_target_oil, test_size=0.2, random_state=0)

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
    
correlation(df_train)## Recherche des paramètres
# In[546]:

## Fit regression model training

regr_gas = ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=10)
regr_oil = ElasticNet(alpha=0.018, l1_ratio=0.7, max_iter=12)

clf_gas = MultiOutputRegressor(regr_gas)
clf_oil = MultiOutputRegressor(regr_oil)

clf_gas.fit(X_train, y_train_gas)
clf_oil.fit(X_train, y_train_oil)

y_predict_gas = clf_gas.predict(X_test)
y_predict_oil = clf_oil.predict(X_test)


# In[547]:

## Metrics training

retour = 'Erreur Moyenne absolue : {}\nErreur Moyenne carré : {}\nR2 : {}'.format(mean_absolute_error(y_test_oil, y_predict_oil),
                mean_squared_error(y_test_oil, y_predict_oil),
                r2_score(y_test_oil, y_predictSVR_oil)               
               )

print(retour)


# In[548]:

## Metrics training

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

retour = 'Erreur Moyenne absolue : {}\nErreur Moyenne carré : {}\nR2 : {}'.format(mean_absolute_error(y_test_gas, y_predict_gas),
                mean_squared_error(y_test_gas, y_predict_gas),
                r2_score(y_test_gas, y_predictSVR_gas),
               )

print(retour)


# In[549]:

## Fit regression model test data

clf_gas.fit(df_data, df_target_gas)
clf_oil.fit(df_data, df_target_oil)

y_predictFin_gas = clf_gas.predict(df_test)
y_predictFin_oil = clf_oil.predict(df_test)


# In[550]:

gascumpred = []
oilcumpred = []

gascumpred = [element[0] for element in y_predictFin_gas]
oilcumpred = [element[0] for element in y_predictFin_oil]

gascumpred_mean = mean_absolute_error(y_test_gas, y_predict_gas)
oilcumpred_mean = mean_absolute_error(y_test_oil, y_predict_oil)

GasCum360_inf = gascumpred - abs(gascumpred_mean)
GasCum360_sup = gascumpred + abs(gascumpred_mean)

OilCum360_inf = oilcumpred - abs(oilcumpred_mean)
OilCum360_sup = oilcumpred + abs(oilcumpred_mean)

GasCum360_inf.tolist
GasCum360_sup.tolist

OilCum360_inf.tolist
OilCum360_sup.tolist


# In[551]:

y_predict_gas[:10]


# In[552]:

y_test_gas[:10]


# In[553]:

## Output

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


# In[554]:

print(output.head(5))

Rappport 1 :

Importation correcte des données
Début du travail exploratoire, affichage du head et description des données
Beaucoup de colonnes sont incomplètes => identification et élimination de ces colonnes
Observation des composantes corrélées.

Rapport 2 :

Elimination des colonnes incomplètes
Exploration des données de sorties
Choix de l'algorithme - DecisionTreeRegressor / AdaBoost

Rapport 3 :

Fin elimination des colonnes incompletes
Fin du formatage des données
Debut mise en place de la cross_validation
Debut mise en place des metrics

Rapport 4 :

Metrics misent en place
Cross validation terminées
Finir de modifier l'algorithme
Trouver une surface acceptable

Utiliser un K-fold
Identifier les composantes principales
Trouver les meilleurs paramètres pour l'algorithme
Imputer les valeurs manquantes
Utiliser un réseau de neuronnes