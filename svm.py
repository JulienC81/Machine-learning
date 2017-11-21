#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
from sklearn import svm
from sklearn.metrics import accuracy_score
answer = 0
clf = svm.SVC(C = 10000.0, kernel = 'rbf')
#features_train = features_train[:len(features_train)/4]
#labels_train = labels_train[:len(labels_train)/4]
t0 = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time() - t0, 3), "s"
pred = clf.predict(features_test)
for i in xrange(len(pred)):
    if pred[i] == 1:
        answer += 1
print len(pred)
print pred
print answer
accuracy = accuracy_score(pred, labels_test)
print accuracy

#########################################################


