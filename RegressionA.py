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

# Define predicted variable
variable_model = "Perimeter"
variable_idx = attributeNames.index(variable_model)
y = X[:, variable_idx]

X_cols = list(range(0, variable_idx)) + list(range(variable_idx + 1, len(attributeNames)))
X = X[:, X_cols]


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

figure(k, figsize=(12, 8))
subplot(1, 2, 1)
semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
xlabel("Regularization factor")
ylabel("Mean Coefficient Values")
grid()
# You can choose to display the legend, but it's omitted for a cleaner
# plot, since there are many attributes
# legend(attributeNames[1:], loc='best')

subplot(1, 2, 2)
title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
loglog(
    lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
)
xlabel("Regularization factor")
ylabel("Squared error (crossvalidation)")
legend(["Train error", "Validation error"])
grid()
show()

# Define weights and linear model
w0 = -0.5
w1 = 0.01
y = w0 + w1 * X

# Perform regression on defined model
model = lm.LinearRegression(fit_intercept=True)
model.fit(X, y)
y_est = model.predict(X)


# Display scatter plot
# figure()
# subplot(2, 1, 1)
# plot(y, y_est, ".")
# xlabel("(true)")
# ylabel("(estimated)")
# subplot(2, 1, 2)
# hist(residual, 40)

# show()
