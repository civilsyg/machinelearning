#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:01:47 2018

@author: ibenfjordkjaersgaard
"""


from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


from projekt2 import *

# Load Matlab data file and extract variables of interest
mat_data = pimaData
X = X
y = pimaData1[:,7]

attributeNames = [
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


# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2
# Maximum number of neighbors
L=100

# K-fold crossvalidation
K = 1000
CV = model_selection.KFold(n_splits=K,shuffle=True)
errors = np.zeros((N,L))
# Initialize variable
#Error_train = np.empty((len(tc),K))
#Error_test = np.empty((len(tc),K))

i=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(i+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()

print('Ran Exercise 7.1.2')



