import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection

from dtuimldmtools import draw_neural_net, train_neural_net

from RegressionA import *
from sklearn.metrics import mean_squared_error
## BASELINE linear regression model ##
K1 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)

# Error variables
test_error = np.empty((K1, 1))
generalization_error = np.empty((K1,1))

i = 0
for di_train, di_test in CV1.split(y):
    # extract training and test set for current CV fold
    y_train = y[di_train]
    y_test = y[di_test]
    
    K2 = 10
    CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
    validation_error = np.empty((K2,1))
    j = 0
    for dj_train, dj_test in CV2.split(y_train):
        # Training on inner fold training data
        mean_train = np.mean(y_train[dj_train])
        y_dj_test = y_train[dj_test]

        # Predict using mean for validation set
        y_pred_validation = np.full_like(y_dj_test, mean_train)

        # Calculate mean squared error for validation set
        validation_error[j] = mean_squared_error(y_dj_test, y_pred_validation)
        j += 1

    # Training on outer fold training data
    mean_train = np.mean(y_train)

    # Predict using mean for validation set
    y_pred_validation = np.full_like(y_test, mean_train)

    # Calculate generalization error
    test_error[i] = mean_squared_error(y_test, y_pred_validation)
    i += 1
generalization_error_total = 0
for i in range(K1):
    generalization_error_total += len(y_test)/N*test_error[i]

print("Generalization error for ", variable_model, ": ",generalization_error_total[0]*sigmaY)

## ANN Model ##
