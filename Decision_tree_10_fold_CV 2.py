#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:18:49 2018

@author: ibenfjordkjaersgaard
"""

# exercise 6.1.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np
from projekt2 import *
from scipy import stats

np.random.seed(2)


# Load Matlab data file and extract variables of interest
mat_data = pimaData

mat_data_values = mat_data.values
N, M = mat_data_values.shape

data = stats.zscore(mat_data_values)

X = np.delete(data,[7],1).squeeze()
y = np.delete(mat_data_values,[0,1,2,3,4,5,6],1).squeeze()

attributeNames = attributeNames = [
    'pregnant',
    'glucose',
    'bloodPressure',
    'skinThickness',
    'bodyMass',
    'pedigreeFunction',
    'age'
    ]
classNames = [ 'Ikke Diabetes','Diabetes']
N, M = X.shape
C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.LeaveOneOut()

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))
Error_dectree = np.empty((K,1))
#max_depth = np.empty((tc,2))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
        
        model_dectree = dtc.fit(X_train, y_train)
        y_dectree = model_dectree.predict(X_test)
        Error_dectree[k] = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)
        
    k+=1

    
f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()


dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)


