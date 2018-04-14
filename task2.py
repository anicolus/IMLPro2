#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.metrics import accuracy_score
import scipy


def print_detail(clf):
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_)

def random_search(estim, param):
    clf = RandomizedSearchCV(estimator=estim, param_distributions=param, n_jobs=8, cv=5, scoring='accuracy', verbose=10, error_score=0, n_iter=200)
    clf.fit(X, y)
    print_detail(clf)

def grid_search(estim, param):
    clf = GridSearchCV(estimator=estim, param_grid=param, n_jobs=8, cv=5, scoring='accuracy', verbose=10, error_score=0)
    clf.fit(X, y)
    print_detail(clf)

def train_print_results(filename, X, y, clf):
    # Prediction phase
    test = np.genfromtxt('test.csv', delimiter=',')
    test = test[1:,:]

    clf.fit(X,y)

    X_pred = test[:,1:]
    y_pred = clf.predict(X_pred)

    n = y_pred.shape[0]

    file = open(filename, 'w')
    file.write("Id,y\n")
    for i in range(0,n):
        file.write(str(int(test[i,0])) + "," + str(int(y_pred[i])) + "\n")

### Main ###

# Import data
train = np.genfromtxt('train.csv', delimiter=',')
train = train[1:,1:]

# Partition data into X and y
y = train[:,0]
X = train[:,1:]

# Train and write results
#train_print_results('outputfile' ,X, y, ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=19, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))


### Below this point is wild testing ###


gbc = GradientBoostingClassifier()
etc = ExtraTreesClassifier(criterion='gini', max_features=None)
rfc = RandomForestClassifier(max_features='log2', criterion='entropy')
knc = KNeighborsClassifier(weights='distance', p=1)

vc = VotingClassifier(estimators=[('gbc', gbc), ('etc', etc), ('rfc', rfc), ('knc', knc)], voting='soft', weights=[1.0, 1.2, 1.1, 1.0])

params = {
'gbc__n_estimators':scipy.stats.randint(low=510, high=540),
'etc__n_estimators':scipy.stats.randint(low=16, high=21),
'rfc__n_estimators':scipy.stats.randint(low=15, high=20),
'knc__n_neighbors':scipy.stats.randint(low=13, high=20)
}

random_search(vc, params)

#random_search(GradientBoostingClassifier(), {'n_estimators':scipy.stats.randint(low=500, high=550), 'learning_rate':[0.1, 1]})


#grid_search(GradientBoostingClassifier(), {'n_estimators':range(520,540), 'learning_rate':[1]})

#grid_search(ExtraTreesClassifier(), {'n_estimators':range(1,21), 'criterion':['gini', 'entropy'], 'max_features':['sqrt','log2',None]})

#random_search(ExtraTreesClassifier(), {'n_estimators':scipy.stats.randint(low=1, high=20), 'criterion':['gini', 'entropy'], 'max_features':['sqrt','log2',None]})
#ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#           max_depth=None, max_features=None, max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=19, n_jobs=1,
#           oob_score=False, random_state=None, verbose=0, warm_start=False)
#0.8895
#{'max_features': None, 'criterion': 'gini', 'n_estimators': 19}


#random_search(RandomForestClassifier(), {'n_estimators':scipy.stats.randint(low=1, high=20), 'criterion':['gini', 'entropy'], 'max_features':['sqrt','log2',None]})
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#            max_depth=None, max_features='log2', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=19, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
# Result: 0.8785 
# {'max_features': 'log2', 'criterion': 'entropy', 'n_estimators': 18}


#random_search(KNeighborsClassifier(), {'n_neighbors':scipy.stats.randint(low=1, high=20), 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute'], 'p':[1,2,3]})

#random_search(SVC(), {'kernel':('linear', 'rbf', 'sigmoid'), 'decision_function_shape':['ovo','ovr'], 'degree':scipy.stats.randint(low=1, high=10), 'C':scipy.stats.randint(low=1, high=10), 'tol':[1e-4]})

#grid_search(SVC(), {'kernel':['poly'],'degree':[2], 'C':[1]})

#grid_search(SVC(), {'kernel':('linear', 'rbf', 'sigmoid', 'poly'), 'decision_function_shape':['ovo','ovr'], 'degree':[2,3,5,7,9], 'C':[1,5,10], 'tol':[1e-4]})

#grid_search(SVR(), {'kernel':('linear', 'rbf'), 'degree':[3,5], 'C':[1,5,10]})

#grid_search(SGDClassifier(), {'loss':('hinge', 'perceptron', 'modified_huber'), 'alpha':[1e-4,1e-5], 'learning_rate':['optimal', 'invscaling'], 'eta0':[0.1,0.01,0.001], 'tol':[1e-5], 'max_iter':[10000]})

#grid_search(LogisticRegression(), {'penalty':['l2'], 'C':[1,5,10], 'solver':['liblinear', 'sag', 'saga'], 'max_iter':[10000]})

#grid_search(LinearSVC(), {'loss':['hinge', 'squared_hinge'], 'tol':[1e-4], 'C':[1,5,10]})
