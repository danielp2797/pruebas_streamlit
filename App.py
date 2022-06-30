import streamlit as st
import numpy as np
import joblib

def make_prediction(x):
    model = joblib.load('model/pipelines/pipe.joblib')
    return model.predict(x)

st.title('Entorno de Pruebas')
st.header('Predecir')

variable1 = st.slider('variable1', min_value=0.0, max_value=100.0, step=1.0)
variable2 = st.slider('variable2', min_value=0.0, max_value=100.0, step=1.0)
variable3 = st.slider('variable3', min_value=0.0, max_value=100.0, step=1.0)
variable4 = st.slider('variable4', min_value=0.0, max_value=100.0, step=1.0)
variable5 = st.slider('variable5', min_value=0.0, max_value=100.0, step=1.0)

if st.button('Predecir'):
    valor = make_prediction(np.array([variable1, variable2, variable3, variable4, variable5]).reshape(1,-1))
    st.success(f'valor previsto: {valor}')

