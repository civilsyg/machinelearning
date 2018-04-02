#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:40:00 2018

@author: ibenfjordkjaersgaard
"""

# exercise 6.3.1

from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy import stats

from Decision_tree_10_fold_CV import Error_dectree
from ANN_class import Error_ANN

Error_dectree = Error_dectree
Error_ANN = Error_ANN
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5


# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
z = (Error_ANN-Error_dectree)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_ANN, Error_dectree),axis=1))
xlabel('ANN   vs.   Decision Tree')
ylabel('Cross-validation error [%]')

show()
