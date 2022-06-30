import streamlit as st
import numpy as np
import joblib

model = joblib.load('model/pipelines/pipe.joblib')

st.title('Entorno de Pruebas')
st.header('Historico')
with st.container():
    st.write("Historico")

    # You can call any Streamlit command, including custom components:
    st.bar_chart(np.random.randn(50, 3))
st.header('Estimacion')

sl_col1, sl_col2  = st.columns(2)
sl_col3, sl_col4 = st.columns(2)
sl_col5, _ = st.columns(2)

variable1 = sl_col1.slider('variable1', min_value=0.0, max_value=10.0, step=1.0, help='selecciona valor de variable')
variable2 = sl_col2.slider('variable2', min_value=0.0, max_value=10.0, step=1.0, help='selecciona valor de variable')
variable3 = sl_col3.slider('variable3', min_value=0.0, max_value=10.0, step=1.0, help='selecciona valor de variable')
variable4 = sl_col4.slider('variable4', min_value=0.0, max_value=10.0, step=1.0, help='selecciona valor de variable')
variable5 = sl_col5.slider('variable5', min_value=0.0, max_value=10.0, step=1.0, help='selecciona valor de variable')

if st.button('Predecir'):
    valor = model.predict(np.array([variable1, variable2, variable3, variable4, variable5]).reshape(1,-1))
    st.success(f'valor previsto: {valor}')

coef = model.get_params()['steps'][1][1].coef_

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(label="Efecto 1", value=int(coef[0]*variable1), delta=0.5,
   delta_color="inverse")
col2.metric(label="Efecto 2", value=int(coef[1]*variable2), delta=23,
   delta_color="inverse")
col3.metric(label="Efecto 3", value=int(coef[2]*variable3), delta=5,
  delta_color="inverse")
col4.metric(label="Efecto 4", value=int(coef[3]*variable4), delta=10,
  delta_color="inverse")
col5.metric(label="Efecto 5", value=int(coef[4]*variable5), delta=33,
     delta_color="inverse")

