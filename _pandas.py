import pandas as pd

"""
Pandas es una herramienta de manipulación de datos de alto nivel, es construido con la biblioteca de Numpy. 
Su estructura de datos más importante y clave en la manipulación de la información es DataFrame, 
el cuál nos va a permitir almacenar y manejar datos tabulados observaciones (filas) y variables (columnas).
"""

print("** PANDAS **")

print("Creando una serie con numeros")
series = pd.Series( [5, 10, 15, 20, 25] )
print(series)

print("Tipo de dato de una serie")
print(type(series))

print("Accediendo a una posicion de la serie")
print(series[3])

print("Creando una serie con caracteres")
cad = pd.Series( ['p', 'l', 'a', 't', 'z', 'i'] )
print(cad)

print("Creando un DataFrame simple")
lst = ['Hola', 'mundo', 'robótico']
df = pd.DataFrame(lst)
print(df)

print("Creando un DataFrame complejo")
data = { 'Nombre': ['Juan', 'Ana', 'Jose', 'Arturo' ], 
        'Edad' : [25, 18, 23, 27], 
        'Pais': ['MX', 'CO', 'BR', 'MX'] }
df = pd.DataFrame(data)
print(df)

print("Mostrando solo algunas columnas del DataFrame")
print(df[[ 'Nombre', 'Pais' ]])

print("Leyendo un archivo CSV con pandas")
data = pd.read_csv('archivos/canciones-2018.csv')
print("Mostrando las primeras 5 filas del archivo")
print(data.head(5))

print("Mostrando solo la columna artists")
print(data.artists)
print("Mostrando la fila 5 de la columna artists")
print(data.artists[5])

print("Mostrando la fila 15 del archivo")
print(data.iloc[15])

print("Mostrando las ultimas filas del archivo")
print(data.tail())

print("Mostrando las dimensiones del archivo")
print(data.shape)

print("Mostrando las columnas del archivo")
print(data.columns)

print("Mostrando informacion estadistica de la columna tempo")
print(data['tempo'].describe())

print("Ordenando el archivo")
print(data.sort_index(axis = 0, ascending = False))