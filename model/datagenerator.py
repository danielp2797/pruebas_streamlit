import numpy as np
from sklearn.datasets import make_regression
np.random.seed(0)

X_test, y_test = make_regression(n_samples=200, n_features=5, noise=100)

np.savetxt("raw/features.csv", X_test, delimiter=',')
np.savetxt("raw/target.csv", y_test, delimiter=',')



