import sys
sys.path.append("/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/DTU/Machine learning/Projekt 1/Script")
from projekt1 import *


sys.path.append("/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/DTU/Machine learning/02450Toolbox_Python/Tools")
import numpy as np
from sklearn import tree

# exercise 5.1.1
# Names of data objects
dataobjectNames = pimaData.reset_index()['index'].values

# Attribute names
attributeNames = ['pregnant','glucose','bloodPressure','skinThickness','bodyMass','pedigreeFunction','age'] 

# Attribute values
pimaData1 = pimaData.values
X = np.delete(pimaData1,7,1)

# Class indices
y = np.asarray(np.mat('3 4 2 3 0 4 3 1 3 2 4 1 3 2 0').T).squeeze()
y = classLabels.values.T
# Class names
classNames = ['Rask', 'Syg']
    
# Number data objects, attributes, and classes
N, M = X.shape
C = len(classNames)

print('Ran Exercise 5.1.1')


# exercise 5.1.2


# requires data from exercise 5.1.1
#from ex5_1_1 import *

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)

print('Ran Exercise 5.1.2')