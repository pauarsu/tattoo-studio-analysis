import pandas as pd

def cargar_datos(ruta):
    """Carga un archivo CSV y devuelve un DataFrame."""
    return pd.read_csv(ruta)

def info_columnas(df):
    """Devuelve un DataFrame con cantidad de nulos y tipo de dato por columna."""
    nulls = df.isnull().sum()
    tipos = df.dtypes.astype(str)
    info = pd.DataFrame({"nulos": nulls, "tipo": tipos})
    return info
def eliminar_columnas(df, columnas):
    """Elimina columnas específicas de un DataFrame."""
    return df.drop(columns=columnas, errors='ignore')

