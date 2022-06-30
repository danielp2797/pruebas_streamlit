import streamlit as st
import numpy as np
import joblib

model = joblib.load('model/pipelines/pipe.joblib')

st.title('Entorno de Pruebas')
st.header('Predecir')

variable1 = st.slider('variable1', min_value=0.0, max_value=100.0, step=1.0)
variable2 = st.slider('variable2', min_value=0.0, max_value=100.0, step=1.0)
variable3 = st.slider('variable3', min_value=0.0, max_value=100.0, step=1.0)
variable4 = st.slider('variable4', min_value=0.0, max_value=100.0, step=1.0)
variable5 = st.slider('variable5', min_value=0.0, max_value=100.0, step=1.0)

if st.button('Predecir'):
    valor = model.predict(np.array([variable1, variable2, variable3, variable4, variable5]).reshape(1,-1))
    st.success(f'valor previsto: {valor}')

coef = model.get_params()['steps'][1][1].coef_

st.metric(label="Efecto 1", value=coef[0]*variable1, delta=0.5,
     delta_color="inverse")
st.metric(label="Efecto 2", value=coef[1]*variable2, delta=23,
     delta_color="inverse")
st.metric(label="Efecto 3", value=coef[2]*variable3, delta=5,
     delta_color="inverse")
st.metric(label="Efecto 4", value=coef[3]*variable4, delta=10,
     delta_color="inverse")
st.metric(label="Efecto 5", value=coef[4]*variable5, delta=33,
     delta_color="inverse")
