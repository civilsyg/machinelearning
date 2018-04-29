# exercise 10.2.1
import sys
sys.path.append('/Users/ibenfjordkjaersgaard/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/machinelearning/Projekt_3')


from matplotlib.pyplot import figure, show, savefig
import numpy as np
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from projekt3 import pimaData, Z,X
from scipy import stats
import pandas as pd 
np.random.seed(2)

#data = pimaData
#X = np.array(pimaData[['pregnant',
#                       'glucose',
#                       'bloodPressure',
#                       'skinThickness',
#                       'bodyMass',
#                       'pedigreeFunction',
#                       'age'
#                       ]])
#
#y = np.array(pimaData[['classVariable']]) # real prediction 

attributeNames = [
    'pregnant',
    'glucose',
    'bloodPressure',
    'skinThickness',
    'bodyMass',
    'pedigreeFunction',
    'age'
    ]

X = Z

N, M = X.shape
C = 2
# Normalize data
#X = stats.zscore(X);

# Load Matlab data file and extract variables of interest

classNames = ['Non diabetes', 'Diabetes']
N, M = X.shape
C = len(classNames)


# Perform hierarchical/agglomerative clustering on data matrix
Method = 'ward' # complete #average # weighted # centroid
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
clsHie= pd.DataFrame(cls)
clsHie.to_csv("clsHie.csv")
figure(1)

clusterplot(X[:,[0,1]], cls.reshape(cls.shape[0],1), y=y)

savefig('hierarchicalScatterPlot.png',dpi=300)
show()


# Display dendrogram
max_display_levels=4
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
savefig('hierachicalDenrogram', dpi = 300)
show()

print('Ran Exercise 10.2.1')