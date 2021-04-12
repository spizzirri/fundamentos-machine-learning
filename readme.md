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

## Matriz de confusión

Los modelos de clasificación son capaces de predecir cuál es la etiqueta correspondiente a cada ejemplo o instancia 
basado en aquello que ha aprendido del conjunto de datos de entrenamiento. Estos modelos necesitan ser evaluados de 
alguna manera y posteriormente comparar los resultados obtenidos con aquellos que fueron entrenados.

Una manera de hacerlo es mediante la matriz de confusión la cual nos permite evaluar el desempeño de un algoritmo de 
clasificación a partir del conteo de los aciertos y errores en cada una de las clases del algoritmo.

|------------------------------------------------|
|   |           |           Prediccion           |
|   |           |  Postivios    |  Negativos     |
|------------------------------------------------|
| O |           | Verdaderos    | Falsos         | 
| b | Positivos | positivos     | Negativos      |
| s |           | (VP)          | (FN)           |
| e |-----------|--------------------------------|
| r |           | Falsos        | Verdaderos     | 
| v | Negativos | Positivos     | Negativos      |
| . |           | (FP)          | (VN)           |
|------------------------------------------------|


### Los verdaderos positivos (VP) 
> son aquellos que fueron clasificados correctamente como positivos como el modelo.
### Los verdaderos negativos (VN) 
> corresponden a la cantidad de negativos que fueron clasificados correctamente como negativos por el modelo.
### Los falsos negativos (FN) 
> es la cantidad de positivos que fueron clasificados incorrectamente como negativos.
### Los falsos positivos (FP) 
> indican la cantidad de negativos que fueron clasificados incorrectamente como negativos.

#### Exactitud

Exactitud = (VP + VN) / Total

#### Tasa de error

Tasa de error = (FP + FN) / Total