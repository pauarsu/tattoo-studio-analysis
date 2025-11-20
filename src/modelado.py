import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def categorizar_horas(horas):
    if horas <= 3:
        return "Corto"
    elif horas <= 6:
        return "Medio"
    else:
        return "Largo"



def dividir_variables(df):
    X = df.drop("Session_Hours_Cat", axis=1)
    y = df["Session_Hours_Cat"]
    return X, y



def dividir_datos(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)



def pipeline_preprocesamiento(X):
    categoricas = X.select_dtypes(include="object").columns
    numericas = X.select_dtypes(include=["int64", "float64"]).columns

    preprocesamiento = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", SimpleImputer(strategy="mean"), numericas)
    ])

    return preprocesamiento


def entrenar_modelo(preprocesamiento, X_train, y_train):

    modelo = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):

    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Corto", "Medio", "Largo"])

    return acc, cm, y_pred


def calcular_matriz_confusion(y_real, y_pred):
    etiquetas = ["Corto", "Medio", "Largo"]
    return confusion_matrix(y_real, y_pred, labels=etiquetas)


def graficar_matriz_confusion(cm):
    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Corto", "Medio", "Largo"],
        yticklabels=["Corto", "Medio", "Largo"]
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()

def predecir(modelo, df):
    df["Predicted_Session"] = modelo.predict(X)
    return df