import pandas as pd

def cargar_datos(ruta):
    return pd.read_csv(ruta)
def eliminar_columnas(df, columnas):
    """Elimina columnas específicas de un DataFrame."""
    return df.drop(columns=columnas, errors='ignore')

