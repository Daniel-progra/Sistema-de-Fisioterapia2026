import streamlit as st
from pathlib import Path
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de Fisioterapia", page_icon="🩺", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Datos fisioterapia 2023 2026.csv"
MODEL_FILE = ROOT / "models" / "modelo_fisioterapia.joblib"

FEATURES = [
    "edad","genero","imc","ocupacion_demanda","tipo_lesion","cronicidad",
    "dolor_inicial_eva","cirugias_previas","comorbilidades_num",
    "rom_inicial_pct","fuerza_inicial_daniels","kinesiofobia_tsk",
    "catastrofismo_pcs","actividad_fisica_previa"
]

st.title("🩺 Sistema de Predicción de Fisioterapia")
st.caption("Modelo de Machine Learning entrenado con 1,200 registros del dataset 2023–2026.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None

df = load_data()
model = load_model()

tab1, tab2, tab3 = st.tabs(["📊 Datos", "🔮 Predicción", "ℹ️ Información"])

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Pacientes", len(df))
    c2.metric("Variables", len(df.columns))
    c3.metric("Resultado = 1", int((df["resultado_tratamiento"]==1).sum()))
    c4.metric("Resultado = 0", int((df["resultado_tratamiento"]==0).sum()))
    st.subheader("Dataset de fisioterapia")
    st.dataframe(df, use_container_width=True, height=500)
    st.subheader("Distribución del resultado")
    st.bar_chart(df["resultado_tratamiento"].value_counts().sort_index())

with tab2:
    if model is None:
        st.error("No se encontró el modelo. Ejecuta: python src/train_model.py")
    else:
        st.subheader("Predicción desde la evaluación inicial")
        st.info("Completa los datos clínicos iniciales. No se utilizan asistencia, cumplimiento de ejercicios ni número de sesiones, porque son variables posteriores al inicio del tratamiento.")
        col1,col2,col3 = st.columns(3)
        with col1:
            edad=st.number_input("Edad",18,100,40)
            genero=st.selectbox("Género",sorted(df.genero.unique()))
            imc=st.number_input("IMC",10.0,60.0,25.0,0.1)
            ocupacion=st.selectbox("Demanda ocupacional",sorted(df.ocupacion_demanda.unique()))
            lesion=st.selectbox("Tipo de lesión",sorted(df.tipo_lesion.unique()))
        with col2:
            cronicidad=st.selectbox("Cronicidad",sorted(df.cronicidad.unique()))
            dolor=st.slider("Dolor inicial EVA",0,10,5)
            cirugias=st.number_input("Cirugías previas",0,10,0)
            comorb=st.number_input("Comorbilidades",0,10,0)
            rom=st.slider("ROM inicial (%)",0,100,60)
        with col3:
            fuerza=st.slider("Fuerza Daniels",0,5,3)
            tsk=st.number_input("Kinesiofobia TSK",0,100,40)
            pcs=st.number_input("Catastrofismo PCS",0,100,30)
            actividad=st.selectbox("Actividad física previa",sorted(df.actividad_fisica_previa.unique()))

        if st.button("🔮 Calcular predicción", type="primary"):
            row=pd.DataFrame([{
                "edad":edad,"genero":genero,"imc":imc,
                "ocupacion_demanda":ocupacion,"tipo_lesion":lesion,
                "cronicidad":cronicidad,"dolor_inicial_eva":dolor,
                "cirugias_previas":cirugias,"comorbilidades_num":comorb,
                "rom_inicial_pct":rom,"fuerza_inicial_daniels":fuerza,
                "kinesiofobia_tsk":tsk,"catastrofismo_pcs":pcs,
                "actividad_fisica_previa":actividad
            }])
            pred=int(model.predict(row)[0])
            prob=float(model.predict_proba(row)[0][pred])
            st.success(f"Resultado predicho: **{pred}**")
            st.metric("Confianza estimada del modelo",f"{prob:.1%}")
            st.warning("Esta predicción es un apoyo estadístico y no sustituye la valoración de un profesional de salud.")

with tab3:
    st.markdown("""
### Sobre el modelo
- Dataset utilizado: `Datos fisioterapia 2023 2026.csv`
- Registros: **1,200**
- Variable objetivo: `resultado_tratamiento`
- Algoritmo: **Random Forest**
- Variables de entrada: datos disponibles en la evaluación inicial.
- El modelo guardado se encuentra en `models/modelo_fisioterapia.joblib`.

### Rendimiento de referencia
Con una división estratificada 80/20 del dataset y semilla 42, el modelo obtuvo aproximadamente **77.9% de exactitud** y **0.881 ROC-AUC** en el conjunto de prueba. Estos valores son una referencia interna, no una validación clínica externa.
""")
