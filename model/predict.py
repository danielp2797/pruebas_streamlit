import joblib


def make_prediction(x):
    model = joblib.load('pipelines/pipe.joblib')
    return model.predict(x)

