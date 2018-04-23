#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:26:42 2018

@author: ibenfjordkjaersgaard
"""

from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, savefig, title
import numpy as np
from scipy import stats
import sys
sys.path.append('/Users/ibenfjordkjaersgaard/Documents/GitHub/machinelearning/02450Toolbox_Python')

#from Variable_selection_in_linear_reg import Error_test_fs_reg#, Error_mean_reg
#from ANN_RegPima import Error_ANN_reg, Error_mean_ANN_reg

Error_reg = np.array([[0.81926178], [0.632515  ], [1.02359398], [0.88060206], [1.04330923]])
#Error_reg = np.squeeze(np.asarray(Error_reg)).T

#Error_mean_reg = Error_mean_reg

Error_ANN_reg = np.array([[0.88469264],
       [1.0421104 ],
       [1.14429892],
       [0.88926079],
       [1.10124869]])


#Error_mean_ANN_reg = Error_mean_ANN_reg

#Error_ANN.shape = (5,1)
## Crossvalidation
# Create crossvalidation partition for evaluation 
K = 5




# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
[tstatistic, pvalue] = stats.ttest_ind(Error_reg,Error_ANN_reg)
# and test if the p-value is less than alpha=0.05. 
z = (Error_reg -Error_ANN_reg)
zb = z.mean()
nu = K-1 #frihedsgrad 
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05
 
zL = zb + sig * stats.t.ppf(alpha/2, nu); # 
zH = zb + sig * stats.t.ppf(1-alpha/2, nu); # tail credibility interval

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_reg , Error_ANN_reg),axis=1))
xlabel('Linear Regression   vs.   ANN')
ylabel('Cross-validation error [%]')
title('Linear Regression vs. ANN')
savefig('Linear_Regression_vs_ANN_reg.png')
show()
