# Fundamentos de Machine Learning

Inteligencia artificial
Aprendizaje automatico
Aprendizaje profundo

## ¿Que es Machine Learning?
Es la capacidad de un algoritmo de adquirir conocimiento a partir de observaciones, aprender de los datos para mejorar, describir y predecir resultados

### Aprendizaje supervisado
Es cuando el algoritmo es previsto de un conjunto de datos pre etiquetado

### Aprendizaje no supervisado
Es cuando el algoritmo no es previsto de un conjunto de datos pre etiquetado

## Librerias de Python
* numpy

> Biblioteca de Python comúnmente usada en la ciencias de datos y aprendizaje automático (Machine Learning). 
Proporciona una estructura de datos de matriz que tiene diversos beneficios sobre las listas regulares.

* pandas

> Pandas es una herramienta de manipulación de datos de alto nivel, es construido con la biblioteca de Numpy. 
Su estructura de datos más importante y clave en la manipulación de la información es DataFrame, 
el cuál nos va a permitir almacenar y manejar datos tabulados observaciones (filas) y variables (columnas).

* ScikitLearn

> Scikit Learn es una biblioteca de Python que está conformada por algoritmos de clasificación, regresión, reducción de la dimensionalidad y clustering. 

Es una biblioteca clave en la aplicación de algoritmos de Machine Learning, 
tiene los métodos básicos para llamar un algoritmo, dividir los datos en entrenamiento 
y prueba, entrenarlo, predecir y ponerlo a prueba.


## Modelos
### ¿Que es la prediccion de datos?

Algoritmos que se definen como "clasificadores" que identifican a que conjunto de categorias pertenecen los datos

Podemos entrenar los datos historicos para que entreguen resultados para
ser aplicados al negocio

### Regresion Lineal

Trabaja con datos del tipo cuantitativos

#### Simple

Algoritmo de aprendizaje supervisado que nos indica la tendencia de un conjunto de datos continuos, modelando las relacion entre una variable dependiente Y y una variable llamada X

Ejemplo Salario / Años de experiencia

  Y
   |
   |   /  Yi = b + mXi
   |  /
   | /
   |/
   |_________ X

* Lo importante es tener variedad de informacion para encontrar una tendencia

#### Multiple

Cuando nuestro problema tiene mas de dos variables se le considera lineal multiple y se trabaja con hiper planos

En nuestro ejemplo seria, ademas de los años de experiencia, tener en cuenta el lenguaje de programacion y/o el pais,

### Regresion Logistica

Si nuestro dato de salida tiene un valor cualitativo utilizamos y aplicamos la regresion logistica

Por ejemplo imaginemos un modelo que debe decir si una persona tiene o no diabetes

   Y
  1 |           / 
    |          /
    |     ____/
    |    /
  0 |___/________ X
    0          1

## Sobreajuste (overfiting)

Es cuando "obligamos" a nuestro modelo a ajustarse a los datos de entrada y salida

## Subajuste (underfiting)

El modelo fallara en el reconocimiento por falta de muestras suficientes. Se entreno con un conjunto de datos muy pequeño