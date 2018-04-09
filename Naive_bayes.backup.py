#Naive Bayes - Egen data



import numpy as np
import xlrd
import numpy as np
from scipy import stats
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show, boxplot,xticks,xlabel,ylabel,legend)
from matplotlib import pyplot
import pandas as pd
from scipy.stats import zscore
from scipy.linalg import svd
from projekt2 import *
np.random.seed(2)


#X = X[:,[2,3,4,5,6]] # extract attributes vi want to use 
y = np.array(pimaData[['classVariable']]) # real prediction 

attributeNames = [
    'pregnant',
    'glucose',
    'bloodPressure',
    'skinThickness',
    'bodyMass',
    'pedigreeFunction',
    'age'
    ]

# Class names
classNames = ['Non diabetic','Diabetic']
N = len(y)
M = len(attributeNames)
C = len(classNames)
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection

# requires data from exercise 4.1.1
#from ex7_2_3 import *
y = y.squeeze()

# Naive Bayes classifier parameters
alpha = 1         # additive parameter (e.g. Laplace correction)
est_prior = True   # uniform prior (change to True to estimate prior from data)

# K-fold crossvalidation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)
besterror_j = 1e100

errors_j = np.zeros(K)*np.nan
errors = np.zeros(K)
Alpha = np.zeros(K)
k=0
for train_index, test_index in CV.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    for train_index_j, test_index_j in CV.split(X_train ,y_train):
        X_train_j = X[train_index_j,:]
        y_train_j = y[train_index_j]
        X_test_j = X[test_index_j,:]
        y_test_j = y[test_index_j]
        print('cv1')
        besterror_j = 1e100

        
        for j in range(1,1001):
            j = j/10000000000000000
            
            nb_classifier_j = MultinomialNB(alpha=j, fit_prior=est_prior)
            nb_classifier_j.fit(X_train_j, y_train_j)
            y_est_prob_j = nb_classifier_j.predict_proba(X_test_j)
            y_est_j = np.argmax(y_est_prob_j,1)
            
            errors_j = np.sum(y_est_j!=y_test_j,dtype=float)/y_test_j.shape[0]
            
            if besterror_j >= errors_j:
                besterror_j = errors_j
                alpha = j
#                print('alpha = {:f}'.format(alpha))
        Alpha[k]=alpha



    print(alpha)

    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=est_prior)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)

    errors[k] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
    k+=1

errorNaiveBayes =errors
# Plot the classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))

print('Ran Exercise 7.2.4')