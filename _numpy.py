import numpy as np

"""
Biblioteca de Python comúnmente usada en la ciencias de datos y aprendizaje automático (Machine Learning). 
Proporciona una estructura de datos de matriz que tiene diversos beneficios sobre las listas regulares.
"""

print("** NUMPY **\n")

print("Creando un arreglo")
arreglo = np.array([10, 30, 20, 4, 30, 51, 7, 2, 4, 40, 100])
print(arreglo)

print("Solo la posicion 4")
print(arreglo[4])

print("Desde la posicion 3 hasta el final")
print(arreglo[3:])

print("Desde la posicion 3 hasta la 7 no inclusive")
print(arreglo[3:7])

print("Desde la posicion 1 hasta el final saltando de a 4")
print(arreglo[1::4])

print("Generando un arreglo de ceros")
print(np.zeros(5))

print("Generando un arreglo de unos")
print(np.ones((4, 5)))

print("Consultando un tipo de dato")
print(type(arreglo))

print("Generando un arreglo de 5 numeros entre el 3 y el 10")
print(np.linspace(3, 10, 5))

print("Genrando una matriz")
matriz = np.array( [['x', 'y', 'z'], ['a', 'c', 'e']])
print(matriz)
type(matriz)

print("Dimensiones de la matriz")
print(matriz.ndim)

print("Ordenando un vector")
print(np.sort(arreglo))

print("Ordenado un vector de tipo complejo que usa cabecera")
cabeceras = [ ('nombre', 'S10' ), ('edad', int) ]
datos = [ ('Juan', 10), ('Maria', 70), ('Javier', 42), ('Samuel', 15) ]
usuarios = np.array(datos, dtype = cabeceras)

print(np.sort(usuarios, order = 'edad' ))

print("Generando un arreglo con 25 numeros consecutivos")
print(np.arange(25))

print("Generando un arreglo con numeros entre el 5 y el 30 no inclusivo")
print(np.arange(5,30))

print("Generando un arreglo con numeros entre el 5 y el 50 saltando de a 5")
print(np.arange(5, 50, 5))

print("Generando una matriz de 3 filas y 5 columnas con el valor 10")
print(np.full( ( 3, 5 ), 10))

print("Generando una matriz diagonal")
print(np.diag ( [1, 3, 9, 10] ))

