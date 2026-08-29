"""
Entrenamiento del modelo con el dataset real de fisioterapia.

Ejecutar desde la carpeta raíz:
    python src/train_model.py

Se predice la columna: resultado_tratamiento
Se excluyen variables posteriores al tratamiento (asistencia, cumplimiento y sesiones)
para evitar fuga de información si la predicción se realiza en la evaluación inicial.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Datos fisioterapia 2023 2026.csv"
MODEL_FILE = ROOT / "models" / "modelo_fisioterapia.joblib"

TARGET = "resultado_tratamiento"
FEATURES = [
    "edad", "genero", "imc", "ocupacion_demanda", "tipo_lesion",
    "cronicidad", "dolor_inicial_eva", "cirugias_previas",
    "comorbilidades_num", "rom_inicial_pct", "fuerza_inicial_daniels",
    "kinesiofobia_tsk", "catastrofismo_pcs", "actividad_fisica_previa"
]

def main():
    df = pd.read_csv(DATA_FILE)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    X, y = df[FEATURES], df[TARGET]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"Exactitud de prueba: {accuracy_score(y_test, pred):.3f}")
    print(f"ROC-AUC de prueba: {roc_auc_score(y_test, prob):.3f}")

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Modelo guardado en: {MODEL_FILE}")

if __name__ == "__main__":
    main()
