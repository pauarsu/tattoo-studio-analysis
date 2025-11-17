#importar las librerias
import pandas as pd  #se encarga del manejo de los datos
import numpy as np #operaciones numericas
import os #manejo de las rutas
from sklearn.impute import SimpleImputer #rellena valores faltantes
import matplotlib.pyplot as plt #para crear graficos y visualizar los datos
import seaborn as sns #visualizacion

#se carga el archivo de datos csv
def cargar_datos(ruta_csv):
    df= pd.read_csv(ruta_csv)
    return df

#resumen general de datos y exploracion completa de los datos
def explorar_datos(df):
    info = {
        "forma": df.shape,
        "columnas": df.columns.tolist(),
        "tipos": df.dtypes,
        "nulos_totales": df.isnull().sum().sum(),
        "nulos_por_columna": df.isnull().sum(),
        "estadisticas_numericas": df.describe(include=[float, int]).T,
        "estadisticas_categoricas": df.describe(include=[object, "category"]).T
    }
    return info

def manejar_nulos(df):

    df = df.copy()

    # Identificar columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    # Imputar medianas
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Rellenar categóricas con "Desconocido"
    for col in cat_cols:
        df[col] = df[col].fillna("Desconocido")

    return df

#limpieza de valores faltantes en el data frame
def valores_nulos(df):
    df = df.copy()

    # Identificar columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    # Imputar medianas
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Rellenar categóricas con "Desconocido"
    for col in cat_cols:
        df[col] = df[col].fillna("Desconocido")

    return df

#rellenar valores con la moda
def imputar_con_moda(df, columna):
    df = df.copy()
    moda = df[columna].mode()[0] # valores mas frecuentes
    df[columna] = df[columna].fillna(moda) #remplaza los valores nulos por los frecuentes
    return df


def contador_duplicados(df):
    return df.duplicated().sum()#cuenta cuantas filas duplicadas hay en la base de datos
def ver_duplicados(df):
    return df[df.duplicated()]#muestra las filas duplicadas

#se eliminaran los duplicados en general o si hay una columna en especifico
def eliminar_duplicados(df, columnas=None):
    df = df.copy()
    if columnas:
        df = df.drop_duplicates(subset=columnas)
    else:
        df = df.drop_duplicates()
    return df

def total_nulos(df):
    total = int(df.isna().sum().sum())
    print("Total de valores nulos:", total)


#mapa de calor de los valores nulos
def mapa_nulos(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="rocket_r")
    plt.title("Mapa de valores faltantes en el dataset")
    plt.xlabel("Variables")
    plt.ylabel("Observaciones")
    plt.show()

def eliminar_nulos(df, columnas=None):
    df = df.copy()
    if columnas:
        df = df.dropna(subset=columnas)
    else:
        df = df.dropna()
    return df

#muestra los valores duplicados mediante una grafica
def visualizar_duplicados(df):
    es_duplicado = df.duplicated()
    conteo = es_duplicado.value_counts()

    etiquetas = ['Únicos', 'Duplicados']
    valores = [conteo.get(False, 0), conteo.get(True, 0)]

    plt.figure(figsize=(5, 5))
    sns.barplot(x=etiquetas, y=valores, palette="pastel")
    plt.title("Distribución de filas duplicadas")
    plt.ylabel("Cantidad de filas")
    plt.show()

#guarda los datos limpios despues de las modificaciones
def guardar_datos(df, ruta_salida):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)


