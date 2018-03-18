import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, subplots, tight_layout, savefig, boxplot
from scipy.linalg import svd
from scipy import stats

pimaData=pd.read_csv('pima-indians-diabetes.data.csv',header=None).rename(
        columns = {0:'pregnant',1: 'glucose',2:'bloodPressure',3:'skinThickness',
                   4:'insulin',5:'bodyMass',6:'pedigreeFunction',7:'age',
                   8:'classVariable'})

pd.value_counts(pimaData['glucose'].values,sort=False)
pd.value_counts(pimaData['glucose'] == 0 )



pimaData = pimaData[pimaData.bloodPressure != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.skinThickness != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.bodyMass != 0] #sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData[pimaData.glucose != 0]
#pimaData = pimaData[pimaData.pedigreeFunction <= 1]#sletter rækker som indholder 0 i denne kolonne
pimaData = pimaData.drop(['insulin'], axis=1) #sletter kolonnen insulin 
#pimaData = pimaData.drop(['pedigreeFunction'], axis=1)
#pimaData = pimaData.drop(['pregnant'], axis=1)
#pimaData = pimaData.drop(['glucose'], axis=1)


PD_corr = pimaData.corr() #Returns the correlation between columns in a DataFrame
PD_desc = pimaData.describe() #Summary statistics for numerical columns
PD_cov  = pimaData.cov() # Return the covariance between columns in a DataFrame 
PD_var  = pimaData.var()  # Return the variance between columns in a DataFrame 




# Extract class names to python list,
# then encode with integers (dict)
classLabels = pimaData['classVariable']
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

#Laver vores DataFrame til en matrix med værdier minus ClassVariable
pimaData1 = pimaData.values
X = np.delete(pimaData1,7,1)



# Compute values of N, M and C.
N = len(y)
M = len(pimaData.columns)
C = len(classNames)


# Subtract mean value from data
Y = (X - np.ones((N,1))*X.mean(0))* 1/(np.std(X))


# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');

savefig('variance_explained')
show()

V = V.T


# Project the centered data onto principal component space
Z = np.matmul(Y, V)


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Pima Indian data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

savefig('PCA')
# Output result to screen
show()

#histogram
 
figure()
titles = ['pregnant','glucose','bloodPressure','skinThickness','bodyMass','pedigreeFunction','age'] 
f,a = subplots(2,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(X[:,idx], bins=50, normed=True)
    ax.plot(X[:,idx],stats.norm.pdf(X[:,idx],loc=np.mean(X[:,idx]),scale=np.std(X[:,idx])),'.',color='red')
    ax.set_title(titles[idx])

tight_layout()
savefig('hist')

#boxplot

figure()
titles = ['pregnant','glucose','bloodPressure','skinThickness','bodyMass','pedigreeFunction','age'] 
f,a = subplots(2,3)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.boxplot(X[:,idx])
    ax.set_title(titles[idx])

tight_layout()
savefig('boxplot')




