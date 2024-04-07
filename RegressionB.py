import numpy as np
from sklearn import model_selection

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
generalization_error_base = []
generalization_error_regular = []
opt_lambdas = []

i = 0
for di_train, di_test in CV1.split(y):
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
    # Calculate generalization error for the base model
    generalization_error_base.append(mean_squared_error(y_test, y_pred_validation)*sigmaY)
    

    # Use function rlr_validate to find optimal lambda with 10-fold cross validation
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, K2)

    # Find the index for optimal lambda value
    index_opt_lambda = 0
    j = 0
    for i in range(l1, l2):
        if i == np.log10(opt_lambda):
            index_opt_lambda = j
            break
        j += 1
        
    generalization_error_regular.append(test_err_vs_lambda[index_opt_lambda]*sigmaY)
    opt_lambdas.append(opt_lambda)

    i += 1

# print generalization error (transformed from standardized to original size)
print("Generalization error for ", variable_model, ": ")
print("Baseline regression model (mean) ", generalization_error_base)
print("Regularization parameters: ", opt_lambdas)
print("Generalization errors for rularization: ", generalization_error_regular)


## paired t-test to get interval and p-values ##
# perform statistical comparison of the models
# compute z with squared error.
# zA = np.abs(y_test - yhatA) ** 2

# # compute confidence interval of model A
# alpha = 0.05
# CIA = st.t.interval(
#     1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA)
# )  # Confidence interval

# # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
# zB = np.abs(y_test - yhatB) ** 2
# z = zA - zB
# CI = st.t.interval(
#     1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
# )  # Confidence interval
# p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value