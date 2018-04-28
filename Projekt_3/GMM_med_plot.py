#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:56:40 2018

@author: ibenfjordkjaersgaard
"""

# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show, savefig
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from toolbox_02450 import clusterplot
from scipy import stats
# Load Matlab data file and extract variables of interest
# Load data from matlab file
from projekt3 import pimaData, X
np.random.seed(2)


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

#X = pimaData.drop(['pregnant','glucose','bloodPressure','skinThickness',
#                   'bodyMass','pedigreeFunction','age',
#                   'classVariable'],axis=1)

#X = stats.zscore(X); #Normalize data
#X = X.values


classNames = ['Non diabetic', 'Diabetic']
N, M = X.shape
C = len(classNames)


# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
reps = 3                # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()

gmm = GaussianMixture(n_components=2, covariance_type=covar_type, n_init=reps).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if covar_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])

    count = 0
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results

figure(1);
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
savefig('BIC_og_AIC_ og_Crossvalidation.png')
show()

print('Ran Exercise 11.1.5')




# Plot results:
#figure(figsize=(14,9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()

## In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
idx = [2,5] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()