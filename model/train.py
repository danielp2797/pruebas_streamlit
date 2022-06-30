from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

X, y = np.genfromtxt("raw/features.csv", delimiter=','), np.genfromtxt("raw/target.csv", delimiter=',')

pipe = Pipeline([('scaler', StandardScaler()), ('svc', LinearRegression())])
pipe.fit(X,y)
joblib.dump(pipe, 'pipelines/pipe.joblib')