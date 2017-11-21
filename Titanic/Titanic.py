
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib nbagg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC


sns.set_style("whitegrid")

train = pd.read_csv(r'C:/Users/COCHE/Documents/machine-learning/Titanic/train.csv')
test = pd.read_csv(r'C:/Users/COCHE/Documents/machine-learning/Titanic/test.csv')

dataset = train.append(test, ignore_index=True)



# Data exploration

#train.head()


# In[ ]:

train.describe()


# In[ ]:

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);


# In[ ]:

def correlation(data):
    correlation = data.corr()
    figure = plt.subplots( figsize =( 12 , 10 ) )
    color_map = sns.diverging_palette( 220 , 10 , as_cmap = True )
    figure = sns.heatmap(
        correlation, 
        cmap = color_map, 
        annot = True, 
        annot_kws = { 'fontsize' : 15 }
    )


# In[ ]:

correlation( train )


# In[ ]:

s = train.Age.isnull().sum()
float(s) / float(891)


# In[ ]:

age_ft = pd.DataFrame()
age_ft["Age"] = dataset.Age.fillna(train.Age.mean())
age_ft.head()


# In[ ]:

#sex_ft = pd.Series(np.where(dataset.Sex == 'male', 1, 0), name = 'Sex')
sex_ft = pd.get_dummies(dataset.Sex, prefix='Sex')
sex_ft.head()


# In[ ]:

dataset_X = pd.concat([age_ft, sex_ft], axis = 1)
dataset_X.head()


# In[ ]:

# Creation des datasets (entrainement, validation et test)

from sklearn.cross_validation import train_test_split

train_valid_x = dataset_X[0:891]
train_valid_y = dataset[:891].Survived
test_x = dataset_X[891:]

train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, train_size = 0.7)

print(train_x.shape)


# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()

parameters = {
    'n_estimators':[4, 6, 9],
    'max_features':['log2', 'sqrt', 'auto'],
    'criterion': ['entropy', 'gini'],
    'max_depth': [2, 3, 5, 10], 
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1,5,8]
}

acc_scorer = make_scorer(accuracy_score)

grid = GridSearchCV(model, parameters, acc_scorer)
grid = grid.fit(train_x, train_y)

clf = grid.best_estimator_

clf.fit(train_x, train_y)

print(clf.score(train_x, train_y), clf.score(train_x, train_y))


# In[ ]:

pred = clf.predict(test_x)
result = dataset[891:].PassengerId
test = pd.DataFrame({'PassengerId': result, 'Survived': pred})
test.shape
test.head()


# In[ ]:



