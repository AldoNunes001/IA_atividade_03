import streamlit as st
import pandas as pd
import joblib

meses = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
}
dias_da_semana = {
    "domingo": 1, "segunda": 2, "terça": 3, "quarta": 4,
    "quinta": 5, "sexta": 6, "sábado": 7
}

st.set_page_config(page_title="Previsão de engajamento", page_icon="🔥")

st.title("Previsão de engajamento")
st.write("Informe os dados solicitados:")
st.write("")

col2, col3 = st.columns(2)

with col2:
    tipo = st.selectbox("Tipo de post:", ["Foto", "Link", "Status", "Video"])
    st.write("")
    pago = st.checkbox("Post Pago")

with col3:
    mes = st.selectbox("Mês", list(meses.keys()))
    dia = st.selectbox("Dia da semana:", list(dias_da_semana.keys()))
    hora = st.slider("Hora do post:", 0, 23, 12)

mes = meses[mes]
dia = dias_da_semana[dia]

encoder = joblib.load('./label_encoder.pkl')  # Carregar o LabelEncoder

tipo_encoded = encoder.transform([tipo])  # Codificar o tipo de post
X = pd.DataFrame({
    'Tipo_encoded': tipo_encoded,
    'Mês': mes,
    'Dia da Semana': dia,
    'Hora': hora,
    'Pago': pago
})

if 'linear_regression_model' not in st.session_state:
    st.session_state['linear_regression_model'] = joblib.load("./linear_regression_le.pkl")

selected_model = st.session_state['linear_regression_model']

result = False
avaliar = st.button("Prever")

if avaliar:
    model_result = selected_model.predict(X.values.reshape(1, -1))
    coments = int(model_result[0][0]) if model_result[0][0] > 0 else 0
    likes = int(model_result[0][1]) if model_result[0][1] > 0 else 0
    shares = int(model_result[0][2]) if model_result[0][2] > 0 else 0
    result = True

st.write("")
st.write("Resultado:")
if result:
    st.info(f"Número de comentários: {coments}")
    st.info(f"Número de likes: {likes}")
    st.info(f"Número de compartilhamentos: {shares}")
