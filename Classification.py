import numpy as np
from sklearn import model_selection
import scipy.stats as st

from ucimlrepo import fetch_ucirepo 
from sklearn import model_selection
from statistics import mode
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

# function for finding most frequent occuring raisin in list (found on https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/)
def most_frequent(List):
    unique, counts = np.unique(List, return_counts=True)
    index = np.argmax(counts)
    return unique[index]

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

# We standardize the data since we have very different scales in our data
mu = np.empty((1, M - 1))
sigma = np.empty((1, M - 1))
mu = np.mean(X[:, 1:], 0)
sigma = np.std(X[:, 1:], 0)

X[:,1:] = (X[:,1:] - mu) / sigma

# Define outer folds
K1 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
# Define the interval for values of lambda (from 10^l1 - 10^l2)
# CHANGE DL FOR BETTER VALUES
l1, l2, dl = -4, 5, 10
lambda_interval = np.logspace(l1, l2, dl)

# Error variables
    
# Base model
opt_value_inner_base = np.empty(K1)
opt_value_base = np.empty(K1)
generalization_error_inner_base = np.empty(K1)
generalization_error_base = np.empty(K1)

# Regularized logistic
models_regular = []
opt_lambda_inner = np.empty(K1)
opt_lambda = np.empty(K1)
generalization_error_inner_regular = np.empty(K1)
generalization_error_regular = np.empty(K1)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))


i = 0
for di_train, di_test in CV1.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[di_train]
    X_test = X[di_test]
    y_train = y[di_train]
    y_test = y[di_test]
    models_j_regular = []

    # Define inner folds
    K2 = 10
    CV2 = model_selection.KFold(n_splits=K2,shuffle=True)
    j = 0
    for dj_train, dj_test in CV2.split(X_train, y_train):
        print("Running iteration (", i, ", ", j,")")
        X_train_j = X[dj_train]
        X_test_j = X[dj_test]
        y_train_j = y[dj_train]
        y_test_j = y[dj_test]
        models_k_regular = []

        ## Regular Logistic model ##
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])
            models_k_regular.append(mdl)
            mdl.fit(X_train_j, y_train_j)
            y_train_est = mdl.predict(X_train_j).T
            y_test_est = mdl.predict(X_test_j).T

            train_error_rate[k] = np.sum(y_train_est != y_train_j) / len(y_train_j)
            test_error_rate[k] = np.sum(y_test_est != y_test_j) / len(y_test_j)

            w_est = mdl.coef_[0]
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

        ## Regular inner ##
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda_inner[j] = lambda_interval[opt_lambda_idx]
        generalization_error_inner_regular[j] = test_error_rate[opt_lambda_idx]
        models_j_regular.append(models_k_regular[opt_lambda_idx])

        ## Base model inner ##
        opt_value_inner_base[j] = most_frequent(y_train_j)
        generalization_error_inner_base[j] = np.sum(opt_value_inner_base[j] != y_test_j) / len(y_test_j)

        j += 1

    ## Base model outer ##
    opt_value_base[i] = most_frequent(opt_value_inner_base)
    generalization_error_base[i] = np.mean(generalization_error_inner_base)

    ## Regular model outer ##
    opt_model_idx = np.argmin(generalization_error_inner_regular)
    opt_lambda[i] = opt_lambda_inner[opt_model_idx]

    mdl = LogisticRegression(penalty="l2", C=1 / opt_lambda[i])
    models_regular.append(mdl)
    mdl.fit(X_train, y_train)
    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T

    train_error = np.sum(y_train_est != y_train) / len(y_train)
    test_error = np.sum(y_test_est != y_test) / len(y_test)
    generalization_error_regular[i] = test_error

    i += 1

## Base model ##
yhat_base = np.ones(N)*most_frequent(opt_value_base)
print("generalization error (base) is: ", generalization_error_base)

## Regular model ##
opt_model_idx = np.argmin(generalization_error_regular)
final_regular_model = models_regular[opt_model_idx]
yhat_regular = final_regular_model.predict(X)
print("generalization error (regular) is: ", generalization_error_regular)
print("optimal lambdas are: ", opt_lambda)
print("Weight values:")
w = final_regular_model.coef_[0]
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w[m], 2)))

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

