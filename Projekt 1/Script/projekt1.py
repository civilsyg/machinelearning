"""
Skript for projekt 1 i indroduktion til machine learning.
Udformet af Mikkel Sinkjær og Iben Fjord Kjærsgaard 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats

plt.style.use('classic') # Set plot theme


## Data reading and removing of uncorrect obervations
pimaData=pd.read_csv('pima-indians-diabetes.data.csv',header=None).rename(
        columns = {0:'pregnant',1: 'glucose',2:'bloodPressure',3:'skinThickness',
                   4:'insulin',5:'bodyMass',6:'pedigreeFunction',7:'age',
                   8:'classVariable'})  # Read the data and name alle the attributes


pimaData = pimaData[pimaData.bloodPressure != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.skinThickness != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.bodyMass != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.glucose != 0]
#pimaData = pimaData[pimaData.pedigreeFunction <= 1]#sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData.drop(['insulin'], axis=1) #sletter kolonnen insulin 

#pimaData = pimaData.drop(['pedigreeFunction'], axis=1)
#pimaData = pimaData.drop(['pregnant'], axis=1)
#pimaData = pimaData.drop(['glucose'], axis=1)

################################3
## Summary statistics

PD_corr = pimaData.corr() #Returns the correlation between columns in a DataFrame

PD_desc = pimaData.describe() #Summary statistics for numerical columns
PD_desc.to_excel('summaryStat.xlsx')

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

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');

plt.savefig('variance_explained')
plt.show()

V = V.T # Transpose V


# Project the centered data onto principal component space
Z = np.matmul(Y, V)


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Pima Indian data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o')
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

plt.savefig('PCA')
# Output result to screen
plt.show()


###############################
##histogram

plt.figure()
plt.titles = ['pregnant','glucose','bloodPressure','skinThickness','bodyMass','pedigreeFunction','age'] 
f,a = plt.subplots(2,3)
a = a.ravel()
for idx,ax in enumerate(a):# for loop makes a histogram of each attribute, and plots a normal distribution over it. 
    ax.hist(X[:,idx], bins=50, normed=True)
    ax.plot(X[:,idx],stats.norm.pdf(X[:,idx],loc=np.mean(X[:,idx]),scale=np.std(X[:,idx])),'.',color='red')
    ax.set_title(plt.titles[idx])

plt.tight_layout()
plt.savefig('hist')


#################################
## Plost Attributes aginst each other.

names = list(pimaData) # list the names of the attributtes


a = 0
plt. figure()

for i in range(len(names)-1):

    for j in range(len(names)-1):
        a += 1
        plt.subplot(7,7,a)
        plt.scatter(X[:,i],X[:,j], s=5, c='black') # Make scatterplot of i'th and j'th attribute

        plt.xticks([]) # Remove x axes
        plt.yticks([]) # Remove y axes
        n = str(i)
plt.savefig('correlation',dpi=300)

plt.show()


#################################
### Correlations matrix

plt.figure()
sns.heatmap(pimaData.corr(), annot=True, fmt=".2f") # Makes color coded correlation matrix 
plt.savefig('ex3.pdf', bbox_inches='tight')



####################################
###boxplot

plt.figure()
titles = ['pregnant','glucose','bloodPressure','skinThickness','bodyMass','pedigreeFunction','age'] 
f,a = plt.subplots(2,3)
a = a.ravel()
for idx,ax in enumerate(a): # Make box plot for each of the attributes
    ax.boxplot(X[:,idx])
    ax.set_title(titles[idx])

plt.tight_layout()
plt.savefig('boxplot')

