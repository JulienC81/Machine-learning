import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(pred, labels_test)


def submitAccuracies():
    return {"acc": round(acc, 3)}

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
