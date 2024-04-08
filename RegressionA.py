from ucimlrepo import fetch_ucirepo 
import numpy as np
from matplotlib.pylab import figure, hist, plot, show, subplot, xlabel, ylabel
import sklearn.linear_model as lm
from sklearn import model_selection
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)

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

## Find the weights for linear model ##
# Define the interval for values of lambda (from 10^l1 - 10^l2)
l1, l2 = -5, 2
lambdas = np.power(10.0, range(l1, l2))

# Use function rlr_validate to find optimal lambda with 10-fold cross validation
k = 10
(
    opt_val_err,
    opt_lambda,
    mean_w_vs_lambda,
    train_err_vs_lambda,
    test_err_vs_lambda,
) = rlr_validate(X, y, lambdas, k)

# Plot for regularization parameter values against estimated generalization error
# figure(k, figsize=(12, 8))
# subplot(1, 2, 1)
# semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
# xlabel("Regularization factor")
# ylabel("Mean Coefficient Values")
# grid()

# subplot(1, 2, 2)
# title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
# loglog(
#     lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
# )
# xlabel("Regularization factor")
# ylabel("Squared error (crossvalidation)")
# legend(["Train error", "Validation error"])
# grid()
# show()

# Extract weigths from the optimal lambda value
index_opt_lambda = 0
j = 0
for i in range(l1, l2):
    if i == np.log10(opt_lambda):
        index_opt_lambda = j
        break
    j += 1
w = []
for i in range(0, len(attributeNames)):
    w.append(mean_w_vs_lambda[i,index_opt_lambda])

# Define weights and linear model
y_reg_values = np.dot(X,w)

# Display scatter plot
# figure()
# subplot(2, 1, 1)
# plot(y, y_reg_values, ".")
# xlabel("(true)")
# ylabel("(estimated)")
# show()

# Print regression model
print("Weight values:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w[m], 2)))

print("Correlation = ", np.corrcoef(y,y_reg_values)[0,1])
print("Generalization test-error: ", test_err_vs_lambda*sigmaY)
print("Generalized train-error: ", train_err_vs_lambda*sigmaY)
