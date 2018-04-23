#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:26:42 2018

@author: ibenfjordkjaersgaard
"""

from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, savefig, title
import numpy as np
from scipy import stats


Error_linear_reg = np.array([[0.81926178], [0.632515  ], [1.02359398], [0.88060206], [1.04330923]])

Error_mean_reg = np.array([[1.02682598],[1.02238967], [1.0090181 ],  [0.98606222], [0.95587375]])


Error_ANN_reg = np.array([[0.88484984], [1.01533648], [1.18072543], [0.89463091],  [1.10124875]])



#Error_ANN.shape = (5,1)
## Crossvalidation
# Create crossvalidation partition for evaluation 
K = 5




# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
[tstatistic, pvalue] = stats.ttest_ind(Error_reg,Error_mean_reg)
# and test if the p-value is less than alpha=0.05. 
z_linear = (Error_linear_reg - Error_mean_reg)
zb_linear = z_linear.mean()
nu = K-1
sig =  (z_linear-zb_linear).std()  / np.sqrt(K-1)
alpha = 0.05

zL_linear = zb_linear + sig * stats.t.ppf(alpha/2, nu);
zH_linear = zb_linear + sig * stats.t.ppf(1-alpha/2, nu);

if zL_linear <= 0 and zH_linear>= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_linear_reg , Error_mean_reg),axis=1))
xlabel('Linear Regression   vs.   subtracting the mean ')
ylabel('Cross-validation error [%]')
title('Linear Regression   vs.   subtracting the mean')
savefig('Linear_Regression_vs_subtracting_the_mean.png')
show()
