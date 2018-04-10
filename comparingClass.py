from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
import scipy

from ANN_Class_Pima import Error_ANN_class
from Decision_tree_opdateret import Error_dectree


[tstatistic, pvalue] = stats.ttest_ind(Error_ANN_class,Error_dectree)
