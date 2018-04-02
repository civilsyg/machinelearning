# ANN regression on Pima 

from matplotlib.pyplot import (figure, plot, subplot, title, xlabel, ylabel, 
                               hold, contour, contourf, cm, colorbar, show,
                               legend)
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection

from projekt2 import *
np.random.seed(2)
plt.style.use('default') # Set plot theme


X = X[:,[1,4]]
y = np.array(pimaData[['classVariable']])

attributeNames = [
#    'pregnant',
    'glucose',
#    'bloodPressure',
#    'skinThickness',
    'bodyMass',
#    'pedigreeFunction',
#    'age'
    ]



classNames = [ 'Ikke Diabetes','Diabetes']
N, M = X.shape
C = len(classNames)

# Parameters for neural network classifier
n_hidden_units = 20      # number of hidden units
n_train = 2             # number of networks trained in each k-fold

# These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
learning_goal = 2.0     # stop criterion 1 (train mse to be reached)
max_epochs = 400        # stop criterion 2 (max epochs in training)

# K-fold CrossValidation (4 folds here to speed up this example)
K = 5
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)*np.nan
Error_ANN = np.empty((K,1))
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]#.astype(int)
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    best_train_error = 1e100
    for i in range(n_train):
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[0, 1], [0, 1]], [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
        if train_error[-1]<best_train_error:
            bestnet.append(ann)
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error
    
    y_est = bestnet[k].sim(X_test)
    y_est = (y_est>.5).astype(int)
    errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
    Error_ANN[k] = 100*(y_est!=y_test).sum().astype(float)/len(y_test)
    k+=1
    

# Print the average classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))


# Display the decision boundary for the several crossvalidation folds.
# (create grid of points, compute network output for each point, color-code and plot).
grid_range = [0, 200, 0, 70]; delta = 1; levels = 100
a = np.arange(grid_range[0],grid_range[1],1)
b = np.arange(grid_range[2],grid_range[3],70/200)
A, B = np.meshgrid(a, b)
values = np.zeros(A.shape)

figure(1,figsize=(18,9))
for k in range(4):
    subplot(2,2,k+1)
    cmask = (y==0).squeeze(); plot(X[cmask,0], X[cmask,1],'.r')
    cmask = (y==1).squeeze(); plot(X[cmask,0], X[cmask,1],'.b')
    title('Model prediction and decision boundary (kfold={0})'.format(k+1))
    xlabel('Feature 1'); ylabel('Feature 2');
    for i in range(len(a)):
        for j in range(len(b)):
            values[i,j] = bestnet[k].sim( np.mat([a[i],b[j]]) )[0,0]
    contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
    contourf(A, B, values, levels=np.linspace(values.min(),values.max(),levels), cmap=cm.RdBu)
    if k==0: colorbar(); legend(['Class A (y=0)', 'Class B (y=1)'])


# Display exemplary networks learning curve (best network of each fold)
figure(2)
bn_id = np.argmax(error_hist[-1,:])
error_hist[error_hist==0] = learning_goal
for bn_id in range(K):
    plot(error_hist[:,bn_id]); xlabel('epoch'); ylabel('train error (mse)'); title('Learning curve (best for each CV fold)')

plot(range(max_epochs), [learning_goal]*max_epochs, '-.')


show()

print('ANN er fucking lort. Du skal aldrig bruge det til noget i livet!!!!!!')



