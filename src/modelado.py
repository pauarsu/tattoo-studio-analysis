import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def dividir_variables(df, objetivo):
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    return X, y


def dividir_datos(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def pipeline_preprocesamiento(X):
    columnas_cat = X.select_dtypes(include=['object']).columns
    columnas_num = X.select_dtypes(include=['int64', 'float64']).columns

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False), columnas_cat),
        ("num", StandardScaler(), columnas_num)
    ])

    return pre



def entrenar_randomforest(preprocesamiento, X_train, y_train):
    modelo = Pipeline([
        ("prep", preprocesamiento),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ))
    ])
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    reporte = classification_report(y_test, pred)
    matriz = confusion_matrix(y_test, pred)
    return acc, reporte, matriz
