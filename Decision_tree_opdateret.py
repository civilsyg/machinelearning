# exercise 6.1.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from sklearn import model_selection, tree
import numpy as np
from projekt2 import pimaData
from scipy import stats

np.random.seed(2)


# Load Matlab data file and extract variables of interest
mat_data = pimaData

mat_data_values = mat_data.values
N, M = mat_data_values.shape

data = stats.zscore(mat_data_values)

X = np.delete(data,[7],1).squeeze()
y = np.delete(mat_data_values,[0,1,2,3,4,5,6],1).squeeze()

attributeNames = attributeNames = [
    'pregnant',
    'glucose',
    'bloodPressure',
    'skinThickness',
    'bodyMass',
    'pedigreeFunction',
    'age'
    ]
classNames = [ 'Ikke Diabetes','Diabetes']
N, M = X.shape
C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))
besterror_t = 1e100
Error_dectree = np.empty((K,1))
Errors_s = np.empty((K,1))
k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    for train_index, test_index in CV.split(X_train,y_train ):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train_t, y_train_t = X[train_index,:], y[train_index]
        X_test_t, y_test_t = X[test_index,:], y[test_index]
    
        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc_t = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc_t = dtc_t.fit(X_train_t,y_train_t.ravel())
            y_est_test_t = dtc_t.predict(X_test_t)
            y_est_train_t = dtc_t.predict(X_train_t)
            
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(y_est_test_t - y_test_t)) / float(len(y_est_test_t))
            misclass_rate_train = sum(np.abs(y_est_train_t - y_train_t)) / float(len(y_est_train_t))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
            
            if Error_test[i,k] < besterror_t:
                max_depth_t = t
                besterror_t = Error_test[i,k].min()
    

    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth_t)
    model_dectree = dtc.fit(X_train, y_train)
    y_dectree = model_dectree.predict(X_test)

    Errors_s[k] = np.power(y_dectree-y_test,2).sum().astype(float)/y_test.shape[0]
    Error_dectree[k] = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)
    break
    k+=1

    
f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])
    
show()

print('Ran Exercise 6.1.2')


dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=9)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)




