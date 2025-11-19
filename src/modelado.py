# src/modelado.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


def dividir_variables(df, objetivo):
    """
    Separa la variable objetivo de las features.
    """
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    return X, y


def dividir_datos(X, y, test_size=0.2, random_state=42):
    """
    Divide el dataset en entrenamiento y prueba.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def pipeline_preprocesamiento(X):
    """
    Crea un pipeline de preprocesamiento:
    - OneHot para categóricas
    - Escalado para numéricas
    """
    columnas_categoricas = X.select_dtypes(include=['object']).columns
    columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns

    preprocesamiento = ColumnTransformer([
        ("cat", OneHotEncoder(drop='first', sparse_output=False), columnas_categoricas),
        ("num", StandardScaler(), columnas_numericas)
    ])

    return preprocesamiento


def entrenar_modelo_regresion(preprocesamiento, X_train, y_train):
    """
    Entrena una regresión lineal.
    """
    modelo = Pipeline([
        ("prep", preprocesamiento),
        ("reg", LinearRegression())
    ])
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_modelo_clasificacion(preprocesamiento, X_train, y_train):
    """
    Entrena una regresión logística para clasificación.
    (Solo si la satisfacción es categórica: 1 a 5 por ejemplo)
    """
    modelo = Pipeline([
        ("prep", preprocesamiento),
        ("clf", LogisticRegression(max_iter=200))
    ])
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_regresion(modelo, X_test, y_test):
    """
    Evalúa la regresión.
    """
    pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return mse, r2


def evaluar_clasificacion(modelo, X_test, y_test):
    """
    Evalúa la clasificación (si aplica).
    """
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    reporte = classification_report(y_test, pred)
    return acc, reporte
