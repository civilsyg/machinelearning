# exercise 8.2.6

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats
import math as m
from projekt2 import *
np.random.seed(2)
plt.style.use('default') # Set plot theme


X = X[:,[2,3,4,5,6]] # extract attributes vi want to use 
y = np.array(pimaData[['glucose']]) # real prediction 

attributeNames = [
#    'pregnant',
    'glucose',
    'bloodPressure',
#    'skinThickness',
    'bodyMass',
#    'pedigreeFunction',
#    'age'
    ]

N, M = X.shape
C = 2

# Normalize data
X = stats.zscore(X);
y = stats.zscore(y)
## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = np.linalg.svd(Y,full_matrices=False)
#V = V.T
##Components to be included as features
#k_pca = 3
#X = X @ V[:,0:k_pca]
#N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
n_train = 2             # number of networks trained in each k-fold
learning_goal = 10     # stop criterion 1 (train mse to be reached)
max_epochs = 150         # stop criterion 2 (max epochs in training)
show_error_freq = 10     # frequency of training status updates

# K-fold crossvalidation
K = 3                   # only five folds to speed up this example
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)*np.nan
errors_j = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))
Error_ANN = np.empty((K,1))
bestnet = list()
bestnet_i = list()
k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    print ('cv2')
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    
    for train_index_j, test_index_j in CV.split(X_train ,y_train):
        X_train_j = X[train_index_j,:]
        y_train_j = y[train_index_j]
        X_test_j = X[test_index_j,:]
        y_test_j = y[test_index_j]
        
        print('cv1')
        
        besterror_j = 1e100
        n_hidden_units = 1
        
        for j in range(2,5):
            print('j = {:d}'.format(j))
            ann_j = nl.net.newff([[-3, 3]]*M, [j, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            test_error_j = ann_j.train(X_train_j, y_train_j.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            
            for i in range(n_train):
                print('Training network {0}/{1}...'.format(i+1,n_train))
                # Create randomly initialized network with 2 layers
                ann_i = nl.net.newff([[-3, 3]]*M, [j, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
                if i==0:
                    bestnet_i.append(ann_i)
                    
                # train network
                train_error_i = ann_i.train(X_train_j, y_train_j.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
                
                if train_error_i[-1] < besterror_j:
                    bestnet_i[k]=ann_i
                    besterror_j = train_error_i[-1]
                    #error_hist[range(len(train_error)),k] = train_error
    
            y_est_j = bestnet_i[k].sim(X_test_j)
            errors_j[k] = np.power(y_est_j-y_test_j,2).sum().astype(float)/y_test_j.shape[0]

            if errors_j[k] < besterror_j:
                n_hidden_units = j
                besterror_j = test_error_j[-1]
    
    print('ude af cv1')
    best_train_error = 1e100
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
            # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
            # train network
        train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error
    
    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test)
    Error_ANN[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    k+=1

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(figsize=(6,7));
subplot(2,1,1); bar(range(0,K),Error_ANN); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure(figsize=(6,7));
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
show()

print('Ran Exercise 8.2.6')

#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer



