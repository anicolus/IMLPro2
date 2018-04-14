#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = np.genfromtxt('train.csv', delimiter=',')
train = train[1:,1:]

# Partition data into X and y
y = train[:,0]
X = train[:,1:]

kf = KFold(n_splits=10, shuffle=False, random_state=False)
coef = np.zeros((3,16))
#coef = np.zeros((2,1800))

# Training phase
clf = SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=5, tol=1e-3, learning_rate='optimal', shuffle=False, random_state=None)
#clf = SVC(random_state=None)

for train_index, test_index in kf.split(X):
    # initialize training set
    X_train = X[train_index]
    y_train = y[train_index]

    # initialize test set
    X_test = X[test_index]
    y_test = y[test_index]
    clf.fit(X_train, y_train)

    coef = coef + clf.coef_

# average coefficients of the classifier
coef = coef / 10
clf.coef_ = coef

# some accuracy testing
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(acc)

""" # Prediction phase
test = np.genfromtxt('test.csv', delimiter=',')
test = test[1:,:]

X_pred = test[:,1:]
y_pred = clf.predict(X_pred)

n = y_pred.shape[0]
print("Id,y")
for i in range(0,n):
    print(str(int(test[i,0])) + "," + str(int(y_pred[i]))) """