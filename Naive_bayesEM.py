#Naive Bayes - Egen data
import random
random.seed(10)
import numpy as np


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
np.random.seed(20)
plt.style.use('default') # Set plot theme


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
alpha = .000001         # additive parameter (e.g. Laplace correction)
est_prior = True   # uniform prior (change to True to estimate prior from data)

# K-fold crossvalidation
K = 20
CV = model_selection.KFold(n_splits=K,shuffle=True)

errors = np.zeros(K)
k=0
for train_index, test_index in CV.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=est_prior)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)

    errors[k] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
    k+=1

# Plot the classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))

print('Ran Exercise 7.2.4')