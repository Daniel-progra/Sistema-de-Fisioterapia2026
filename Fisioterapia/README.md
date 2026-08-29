# Sistema de Fisioterapia - Predicción

Proyecto base para una aplicación local de predicción usando Python, Streamlit y Machine Learning.

## 1. Estructura

```text
sistema-fisioterapia-prediccion/
├── .streamlit/
│   └── config.toml
├── data/
│   └── Datos fisioterapia 2023 2026.csv
├── models/
│   └── modelo_fisioterapia.joblib
├── src/
│   ├── db_connector.py
│   └── train_model.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

## 2. Instalar Python

Se recomienda Python 3.11 o 3.12.

Comprueba la instalación:

```bash
python --version
```

## 3. Abrir la carpeta en CMD

```bash
cd ruta\sistema-fisioterapia-prediccion
```

## 4. Crear entorno virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 5. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 6. Colocar el dataset

Guarda tu archivo CSV exactamente como:

```text
data/Datos fisioterapia 2023 2026.csv
```

El script de entrenamiento espera una columna objetivo llamada:

```text
resultado
```

Si tu columna objetivo tiene otro nombre, modifica `TARGET` en `src/train_model.py`.

## 7. Entrenar el modelo

Desde la raíz del proyecto:

```bash
python src/train_model.py
```

Esto generará:

```text
models/modelo_fisioterapia.joblib
```

## 8. Ejecutar la aplicación

```bash
streamlit run app.py
```

Streamlit mostrará una dirección local, normalmente:

```text
http://localhost:8501
```

Ábrela en Chrome.

## 9. Supabase (opcional)

Si vas a conectar una base de datos Supabase:

1. Copia `.env.example` y renómbralo a `.env`.
2. Introduce `SUPABASE_URL` y `SUPABASE_KEY`.
3. No compartas el archivo `.env` ni publiques sus claves.

## Importante

El archivo `.joblib` NO debe inventarse como si fuera un modelo entrenado. Debe generarse ejecutando `src/train_model.py` con tu dataset real.

Si tu CSV tiene columnas diferentes, adapta las variables del script de entrenamiento y la interfaz de `app.py`.
