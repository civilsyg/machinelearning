#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:36:20 2018

@author: ibenfjordkjaersgaard
"""

import sys
sys.path.append('/Users/ibenfjordkjaersgaard/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Semester 4/Machine learning og data mining/02450Toolbox_Python/Tools')

# exercise 6.2.1
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, axes
from mpl_toolkits import mplot3d
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import statsmodels.formula.api as sm
from scipy import stats


from projekt2 import *

np.random.seed(2)

# Load Matlab data file and extract variables of interest
mat_data = pimaData

mat_data_values = mat_data.values
N, M = mat_data_values.shape

data = stats.zscore(mat_data_values)

X = np.delete(data,[1,7],1).squeeze()

y = np.delete(data,[0,2,3,4,5,6,7],1).squeeze()

attributeNames = attributeNames = [
    'pregnant',
    'bloodPressure',
    'skinThickness',
    'bodyMass',
    'pedigreeFunction',
    'age'
    ]

N, M = X.shape


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs_reg = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

# bruges til at gve de 5 Squared error 
SE = np.zeros((K,2))
k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    # forward 
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    # bestem squared error
    SE[k,0] = k+1
    SE[k,1] = loss_record[-1]
    
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs_reg[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
       
        
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1
    




# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs_reg.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs_reg.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f= np.array(np.where(SE[:,1] == SE.min(0)[1])).flatten()+1# cross-validation fold to inspect with the lowest SE
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m_reg = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    model = sm.OLS(y, X[:,ff]).fit()    
    y_est_reg= m_reg.predict(X[:,ff])
    residual=y-y_est_reg
        
    print('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    print(SE)
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
        subplot(2,np.ceil(len(ff)/2.0),i+1)
        plot(X[:,ff[i]],residual,'.')
        xlabel(attributeNames[ff[i]])
        ylabel('residual error')


show()

w0_ny = m.intercept_
coef = m.coef_

results = model.summary()
print('summary af den med det laveste Squared error{}'.format(results))



X = np.delete(data,[0,1,3,5,6,7],1).squeeze()

y = np.delete(data,[0,2,3,4,5,6,7],1).squeeze()

m_reg = lm.LinearRegression(fit_intercept=True, normalize = True).fit(X, y)
y_reg = m_reg.predict(X)
model = sm.OLS(y, X).fit()  
results = model.summary()

results





