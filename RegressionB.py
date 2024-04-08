#Libaries:
import numpy as np
from sklearn import model_selection
import scipy.stats as st

from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn import model_selection
from dtuimldmtools import draw_neural_net, train_neural_net
import torch
from dtuimldmtools import rlr_validate

# fetch data 
raisin = fetch_ucirepo(id=850) 

# data (as pandas dataframes) 
X1 = raisin.data.features 
y = raisin.data.targets 
N, M = X1.shape

# Attribute Names
attributeNames = X1.columns.values.tolist()
# Making classlabels ones & zeros
classLabels = np.asarray(y.Class)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

y = np.asarray([classDict[value] for value in classLabels])


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

# Choose variable to predict
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

############################## Data Initialisation Complete ################################
###############################     Also standardisation   #################################

# Define outer folds
K1 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)

# Define the interval for values of lambda (from 10^l1 - 10^l2)
l1, l2 = -5, 2
lambdas = np.power(10.0, range(l1, l2))

# Neural Network configuration
n_hidden_units_options = range(1, 4)  # Different h-values
n_replicates = 1  # Replicates for the NN training
max_iter = 10000  # Maximum iterations for NN training

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

    ##          BASE            ##
    validation_error = np.empty((K2,1))
    j = 0
    for dj_train, dj_test in CV2.split(y_train):
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
    
    ##            ANN             ##
    #Model-Definition
    for n_hidden_units in n_hidden_units_options:
        model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
        j = 0
        for k, (dj_train, dj_test) in CV2.split(y_train):
            errors_ANN = []
            print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K2))

            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[dj_train, :])
            y_train = torch.Tensor(y[dj_train])
            X_test = torch.Tensor(X[dj_test, :])
            y_test = torch.Tensor(y[dj_test])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

        print("\n\tBest loss: {}\n".format(final_loss))
        # Determine estimated class labels for test set
        y_test_est = net(X_test)

        # Determine errors
        se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        errors_ANN.append(mse)  # store error rate for current CV fold    
    
        # Average error across inner folds for this NN configuration
        avg_ANN_error = np.mean(errors_ANN)
        print(f"Average error for ANN with {n_hidden_units} hidden units: {avg_ANN_error}")

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

