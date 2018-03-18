#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:10:59 2018

@author: ibenfjordkjaersgaard
"""
#skal den plotte kun to prikker i det første plot? 
from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from projekt2 import *

# Load Matlab data file and extract variables of interest
mat_data = pimaData
X = X
y = pimaData[['classVariable']].squeeze()

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


# Simple holdout-set crossvalidation
test_proportion = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)



# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,2], X_train[class_mask,4], styles[c], color ='black')


# K-nearest neighbors
K=26

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=1

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist);
knclassifier.fit(X_train, y_train);
y_est = knclassifier.predict(X_test);


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,2], X_test[class_mask,4], styles[c], markersize=10, color = 'blue')
    plot(X_test[class_mask,3], X_test[class_mask,4], 'kx', markersize=8, color = 'red')
title('Synthetic data classification - KNN');
xlabel('{0}'.format(attributeNames[2])); 
ylabel('{0}'.format(attributeNames[4]));

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()

print('Ran Exercise 7.1.1')