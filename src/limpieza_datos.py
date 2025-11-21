import pandas as pd
import os

def cargar_datos(ruta):
    return pd.read_csv(ruta)
def eliminar_columnas(df, columnas):
    return df.drop(columns=columnas, errors='ignore')
def guardar_datos(df, ruta):
    df.to_csv(ruta, index=False)


