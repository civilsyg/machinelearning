# Binarizing data and splitting diabetes in one out of 

import sys
sys.path.append("02450Toolbox_Python/Tools")
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
from similarity import binarize2
from writeapriorifile import WriteAprioriFile
import pandas as pd
from projekt3 import X,names,y
data= np.concatenate((X, y), axis=1)
[data,Names]=binarize2(data,names)

WriteAprioriFile(data,titles=None,filename="pimaBinarize.txt")
#WriteAprioriFile(X,titles=Names,filename="pimaBinarize.txt")