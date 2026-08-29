import streamlit as st
from pathlib import Path
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción de Fisioterapia",
    page_icon="🩺",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "modelo_fisioterapia.joblib"

st.title("🩺 Sistema de Predicción de Fisioterapia")
st.write("Aplicación local desarrollada con Streamlit y Machine Learning.")

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)

model = load_model()

if model is None:
    st.warning(
        "Todavía no existe el modelo entrenado. "
        "Coloca tu dataset en data/ y ejecuta: "
        "`python src/train_model.py`"
    )
else:
    st.success("Modelo cargado correctamente.")

st.divider()

st.subheader("📊 Consulta de datos")
data_file = ROOT / "data" / "Datos fisioterapia 2023 2026.csv"

if data_file.exists():
    try:
        df = pd.read_csv(data_file)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Registros: {len(df)} | Columnas: {len(df.columns)}")
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
else:
    st.info("No se encontró el archivo CSV del dataset.")

st.divider()
st.subheader("🔮 Predicción")

if model is not None and data_file.exists():
    try:
        df = pd.read_csv(data_file)
        target = "resultado"

        if target in df.columns and len(df) > 0:
            st.write("Selecciona un registro del dataset para realizar una predicción.")
            idx = st.number_input(
                "Número de registro",
                min_value=0,
                max_value=max(0, len(df) - 1),
                value=0,
                step=1
            )

            row = df.drop(columns=[target]).iloc[[idx]]
            if st.button("Realizar predicción"):
                prediction = model.predict(row)[0]
                st.success(f"Resultado predicho: **{prediction}**")
        else:
            st.info("Para activar esta sección, el CSV debe tener una columna llamada 'resultado'.")
    except Exception as e:
        st.error(f"No fue posible realizar la predicción: {e}")
else:
    st.info("Entrena y guarda el modelo para habilitar las predicciones.")
