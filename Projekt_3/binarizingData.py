# Binarizing data and splitting diabetes in one out of 

import sys
sys.path.append("02450Toolbox_Python/Tools")
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
from similarity import binarize2
from writeapriorifile import WriteAprioriFile
import pandas as pd
# Load Matlab data file and extract variables of interest
## Data reading and removing of uncorrect obervations
pimaData=pd.read_csv('pima-indians-diabetes.data.csv',header=None).rename(
        columns = {0:'pregnant',1: 'glucose',2:'bloodPressure',3:'skinThickness',
                   4:'insulin',5:'bodyMass',6:'pedigreeFunction',7:'age',
                   8:'classVariable'})  # Read the data and name alle the attribute
data = np.squeeze(pimaData.values)
ySyg = data[:,[8]] 
yRask = abs(ySyg-1) # laver en rækker hver 1=rask og 0=syg
data = np.delete(data,8,1)
attributeNames = list(pimaData)

[data,Names]=binarize2(data,attributeNames)
X = np.concatenate((data, yRask,ySyg), axis=1)
WriteAprioriFile(X,titles=None,filename="pimaBinarize.txt")