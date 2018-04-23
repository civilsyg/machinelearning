#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:22:53 2018

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
[tstatistic, pvalue] = stats.ttest_ind(Error_ANN_reg ,Error_mean_reg )
# and test if the p-value is less than alpha=0.05. 
z_ANN_reg = (Error_ANN_reg - Error_mean_reg)
zb_ANN_reg= z_ANN_reg.mean()
nu = K-1
sig =  (z_ANN_reg-zb_ANN_reg).std()  / np.sqrt(K-1)
alpha = 0.05

zL_ANN_reg= zb_ANN_reg + sig * stats.t.ppf(alpha/2, nu);
zH_ANN_reg = zb_ANN_reg + sig * stats.t.ppf(1-alpha/2, nu);

if zL_ANN_reg <= 0 and zH_ANN_reg>= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_ANN_reg , Error_mean_reg),axis=1))
xlabel('ANN Regression   vs.   Subtracting the mean ')
ylabel('Cross-validation error [%]')
title('ANN Regression   vs.   Subtracting the mean')
savefig('ANN_vs_subtracting_the_mean.png')
show()
