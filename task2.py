#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

train = np.genfromtxt('train.csv', delimiter=',')
train = train[1:,1:]

# Partition data into X and y
y = train[:,0]
X = train[:,1:]

# GridSearchCV
# SVC
parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'decision_function_shape':['ovo','ovr'], 'degree':[3,5,7,9], 'C':[1,5,10], 'tol':[1e-4]}
svc = SVC()
clf = GridSearchCV(estimator=svc, param_grid=parameters, n_jobs=8, cv=10, scoring='accuracy', verbose=10)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)

""" # SVR (Problem with multiclass)
parameters = {'kernel':('linear', 'rbf'), 'degree':[3,5], 'C':[1,5,10]}
svr = SVR()
clf = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, cv=10, scoring='accuracy', verbose=10)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_) """

""" # SGDClassifier
parameters = {'loss':('hinge', 'perceptron', 'modified_huber'), 'alpha':[1e-4,1e-5], 'learning_rate':['optimal', 'invscaling'], 'eta0':[0.1,0.01,0.001], 'tol':[1e-5], 'max_iter':[10000]}
sgd = SGDClassifier()
clf = GridSearchCV(estimator=sgd, param_grid=parameters, n_jobs=8, cv=10, scoring='accuracy', verbose=10)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_) """

""" # LogisticRegression
parameters = {'penalty':['l2'], 'C':[1,5,10], 'solver':['liblinear', 'sag', 'saga'], 'max_iter':[10000]}
logr = LogisticRegression()
clf = GridSearchCV(estimator=logr, param_grid=parameters, n_jobs=8, cv=10, scoring='accuracy', verbose=10)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_) """

""" # LinearSVC
parameters = {'loss':['hinge', 'squared_hinge'], 'tol':[1e-4], 'C':[1,5,10]}
svc = LinearSVC()
clf = GridSearchCV(estimator=svc, param_grid=parameters, n_jobs=8, cv=10, scoring='accuracy', verbose=10)
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_) """

# Prediction phase
""" test = np.genfromtxt('test.csv', delimiter=',')
test = test[1:,:]

X_pred = test[:,1:]
y_pred = clf.predict(X_pred)

n = y_pred.shape[0]
print("Id,y")
for i in range(0,n):
    print(str(int(test[i,0])) + "," + str(int(y_pred[i]))) """