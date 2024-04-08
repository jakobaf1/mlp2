import numpy as np
from sklearn import model_selection
import scipy.stats as st

from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import model_selection

from dtuimldmtools import rlr_validate

# fetch dataset 
raisin = fetch_ucirepo(id=850) 

# data (as pandas dataframes) 
X1 = raisin.data.features 
y = raisin.data.targets 
N, M = X1.shape

classLabels = np.asarray(y.Class)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))
attributeNames = X1.columns.values.tolist()

y = np.asarray([classDict[value] for value in classLabels])
C = len(classNames)

# translating data to matrix format
X = np.empty((N, 7))
X[:,0] = np.asarray(X1.Area)
X[:,1] = np.asarray(X1.MajorAxisLength)
X[:,2] = np.asarray(X1.MinorAxisLength)
X[:,3] = np.asarray(X1.Eccentricity)
X[:,4] = np.asarray(X1.ConvexArea)
X[:,5] = np.asarray(X1.Extent)
X[:,6] = np.asarray(X1.Perimeter)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

# Define predicted variable
temp = y.copy()
variable_model = "Area"
if (variable_model != "Class"):
    variable_idx = attributeNames.index(variable_model)
    y = X[:,variable_idx].copy()
    X[:,variable_idx] = temp
    attributeNames[variable_idx] = "Class"
# We standardize the data since we have very different scales in our data
mu = np.empty((1, M - 1))
sigma = np.empty((1, M - 1))
mu = np.mean(X[:, 1:], 0)
muY = np.mean(y,0)
sigmaY = np.std(y,0)
sigma = np.std(X[:, 1:], 0)

X[:,1:] = (X[:,1:] - mu) / sigma
y = (y-muY)/sigmaY

# Define outer folds
K1 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
# Define the interval for values of lambda (from 10^l1 - 10^l2)
l1, l2 = -5, 2
lambdas = np.power(10.0, range(l1, l2))

# Error variables
# Base model
generalization_error_base = np.empty(K1)
mean_train = np.empty((K1,K1))
base_models = np.empty((K1,1))

# Regular model
opt_lambdas = np.empty(K1)
w_rlr = np.empty((M, K1))
Error_test_rlr = np.empty(K1)

i = 0
for di_train, di_test in CV1.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[di_train]
    X_test = X[di_test]
    y_train = y[di_train]
    y_test = y[di_test]

    # Define inner folds
    K2 = 10
    CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
    validation_error = np.empty((K2,1))
    j = 0
    for dj_train, dj_test in CV2.split(X_train, y_train):
        ## BASE ##
        # Training on inner fold training data
        mean_train[i][j] = np.mean(y_train[dj_train])
        y_dj_test = y_train[dj_test]

        # Predict using mean for validation set
        y_pred_validation = np.full_like(y_dj_test, mean_train[i][j])
        # Calculate mean squared error for validation set
        validation_error[j] = mean_squared_error(y_dj_test, y_pred_validation)
        j += 1

    # Add the results of the inner fold
    base_models[i] = np.mean(mean_train[i])
    # Calculate generalization error for the base model
    generalization_error_base[i] = np.mean(validation_error)*sigmaY
    
    ## ANN ## 
    


    ## REGULAR ##
    # Use function rlr_validate to find optimal lambda with 10-fold cross validation
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, K2)
    opt_lambdas[i] = opt_lambda
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, i] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    Error_test_rlr[i] = (np.square(y_test - X_test @ w_rlr[:, i]).sum(axis=0) / y_test.shape[0])*sigmaY

    i += 1

# print generalization error (transformed from standardized to original size)
print("Generalization error for ", variable_model, ": ")
print("Baseline regression model (mean) ", generalization_error_base)
print("Regularization parameters: ", opt_lambdas)
print("Generalization errors for regularization: ", Error_test_rlr)

# Define models by averaging the results of cross-validation
#Base
y_est_base = np.mean(base_models[0])
yhat_base = np.ones(N)*y_est_base
# print("y_est_base = ", y_est_base*sigmaY+muY)

#Regular
w_final = np.empty(M)
for i in range(len(w_rlr[:])):
    w_final[i] = np.mean(w_rlr[i,:])
yhat_regular = np.dot(X,w_final)

## paired t-test to get interval and p-values ##
# perform statistical comparison of the models
alpha = 0.05
# compute z with squared error for each model:
# REGULARIZED #
zR = np.abs(y - yhat_regular) ** 2
# BASE #
zB = np.abs(y - yhat_base) ** 2
# ANN #
# zA =

# compute confidence interval of model A
# CIA = st.t.interval(
#     1 - alpha, df=len(zR) - 1, loc=np.mean(zR), scale=st.sem(zR)
# )  # Confidence interval regularized regression model

# Compute confidence interval and p-value of Null hypothesis #

# Base vs. Regularized #
z = zR - zB
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval Baseline regression model
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

print("Confidence interval Base vs. Regularized: ", CI)
print("p-value Base vs. Regularized: ", p)

# Base vs. ANN #
# z = zA - zB
# CI = st.t.interval(
#     1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
# )  # Confidence interval Baseline regression model
# p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

# print("Confidence interval Base vs. ANN: ", CI)
# print("p-value Base vs. ANN: ", p)

# Regularized vs. ANN #
# Base vs. Regularized #
# z = zR - zA
# CI = st.t.interval(
#     1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
# )  # Confidence interval Baseline regression model
# p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

# print("Confidence interval ANN vs. Regularized: ", CI)
# print("p-value ANN vs. Regularized: ", p)

