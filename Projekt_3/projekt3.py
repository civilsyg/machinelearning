#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:29:36 2018

@author: ibenfjordkjaersgaard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:40:11 2018

@author: ibenfjordkjaersgaard
"""

"""
Skript for projekt 1 i indroduktion til machine learning.
Udformet af Mikkel Sinkjær og Iben Fjord Kjærsgaard 
"""
import numpy as np
import pandas as pd
from scipy.linalg import svd

#plt.style.use('bmh') # Set plot theme

import sys
sys.path.append('/Users/ibenfjordkjaersgaard/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/machinelearning/Projekt_3')

## Data reading and removing of uncorrect obervations
pimaData=pd.read_csv('pima-indians-diabetes.data.csv',header=None).rename(
        columns = {0:'pregnant',1: 'glucose',2:'bloodPressure',3:'skinThickness',
                   4:'insulin',5:'bodyMass',6:'pedigreeFunction',7:'age',
                   8:'classVariable'})  # Read the data and name alle the attributes


pimaData = pimaData[pimaData.bloodPressure != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.skinThickness != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.bodyMass != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.glucose != 0]
pimaData = pimaData[pimaData.insulin != 0]
pimaData = pimaData[pimaData.pedigreeFunction <= 1]#sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData.drop(['insulin'], axis=1) #sletter kolonnen insulin 

################################
## Summary statistics

PD_corr = pimaData.corr() #Returns the correlation between columns in a DataFrame

PD_desc = pimaData.describe() #Summary statistics for numerical columns


PD_cov  = pimaData.cov() # Return the covariance between columns in a DataFrame 

PD_var  = pimaData.var()  # Return the variance between columns in a DataFrame 


############################
## PCA 

# Extract class names to python list,
# then encode with integers (dict)
classLabels = pimaData['classVariable']
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

# Convert data frame to a matrix and leave out ClassVariable
pimaData1 = pimaData.values
X = np.delete(pimaData1,7,1)



# Compute values of N, M and C.
N = len(y)
M = len(pimaData.columns)
C = len(classNames)


# Standardize the data
Y = (X - np.ones((N,1))*X.mean(0))* 1/(np.std(X))


# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 



V = V.T # Transpose V


# Project the centered data onto principal component space
Z = np.matmul(Y, V)




#################################
## Plost Attributes aginst each other.

names = list(pimaData) # list the names of the attributtes



