"""
Entrenamiento del modelo de predicción de fisioterapia.

Ejecutar desde la carpeta raíz:
    python src/train_model.py

El modelo se guarda en:
    models/modelo_fisioterapia.joblib
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Datos fisioterapia 2023 2026.csv"
MODEL_FILE = ROOT / "models" / "modelo_fisioterapia.joblib"

# Variables esperadas. Puedes adaptar estas columnas a tu dataset real.
TARGET = "resultado"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {DATA_FILE}\n"
            "Coloca tu archivo CSV con ese nombre dentro de data/."
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(
            f"El CSV debe contener la columna objetivo '{TARGET}'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipeline.fit(X, y)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)

    print(f"Modelo guardado correctamente en: {MODEL_FILE}")

if __name__ == "__main__":
    main()
