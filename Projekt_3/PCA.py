## PCA 
import projekt3, 
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