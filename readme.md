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

## ¿Que es el aprendizaje profundo?
Subcategoria del ML que crea diferentes niveles de abstraccion que representa
los datos

## Red neuronal convolucional
Modelan de forma consecutiva pequeñas piezas de informacion, al final combinan
informacion en las capaz mas profundas de la red

### ¿Que es ReLU?
Funcion de activicion, permite el paso de todos los valores positivos sin cambiarlos pero asigna todos los valores negativos a 0

```
         |    /
         |   /
         |  /
         | /
         |/        
---------|---------
           0
```

#### Tensor Flow
Biblioteca de codigo abierto desarrollado por Google capaz de construir y 
entrenar redes neuronales


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

#### _Simple_

Algoritmo de aprendizaje supervisado que nos indica la tendencia de un conjunto de datos continuos, modelando las relacion entre una variable dependiente Y y una variable llamada X

Ejemplo Salario / Años de experiencia
```
  Y
   |
   |   /  Yi = b + mXi
   |  /
   | /
   |/
   |_________ X

```

* Lo importante es tener variedad de informacion para encontrar una tendencia

#### _Multiple_

Cuando nuestro problema tiene mas de dos variables se le considera lineal multiple y se trabaja con hiper planos

En nuestro ejemplo seria, ademas de los años de experiencia, tener en cuenta el lenguaje de programacion y/o el pais,

### Regresion Logistica

Si nuestro dato de salida tiene un valor cualitativo utilizamos y aplicamos la regresion logistica

Por ejemplo imaginemos un modelo que debe decir si una persona tiene o no diabetes

```
   Y
  1 |           / 
    |          /
    |     ____/
    |    /
  0 |___/________ X
    0          1
```

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

```
|------------------------------------------------|
|   |           |           Prediccion           |
|   |           |  Postivios     |  Negativos    |
|------------------------------------------------|
| O |           | Verdaderos    | Falsos         | 
| b | Positivos | positivos     | Negativos      |
| s |           | (VP)          | (FN)           |
| e |-----------|--------------------------------|
| r |           | Falsos        | Verdaderos     | 
| v | Negativos | Positivos     | Negativos      |
| . |           | (FP)          | (VN)           |
|------------------------------------------------|
```

### Los verdaderos positivos (VP) 
> son aquellos que fueron clasificados correctamente como positivos como el modelo.
### Los verdaderos negativos (VN) 
> corresponden a la cantidad de negativos que fueron clasificados correctamente como negativos por el modelo.
### Los falsos negativos (FN) 
> es la cantidad de positivos que fueron clasificados incorrectamente como negativos.
### Los falsos positivos (FP) 
> indican la cantidad de negativos que fueron clasificados incorrectamente como negativos.

#### _Exactitud_

Exactitud = (VP + VN) / Total

#### _Tasa de error_

Tasa de error = (FP + FN) / Total

## Arboles de decision
Es una forma grafica y analitica que presenta sucesos y sus posibles consecuencias

```
              A
           /     \
          P       P
        /   \    /  \  
       D     P   D   D
           /   \
          D     D
```
### Ventajas

* Claridad en los datos
* Tolerante al ruido y valores faltantes
* Las reglas extraidas permiten hacer predicciones

### Desventajas

* Criterio de division es deficiente
* Sobreajuste
* Ramas poco significativas

#### _¿Como se divide un arbol de decision_
* Ganancia de informacion
* Dividir en pequeños arboles

#### _Optimizacion del arbol_
* Evitar sobreajuste
* Seleccion de atributos que permitan dividir mejor
* Evitar campos nulos

## K-mean

Es un algoritmo que crea K grupos a partir de un conjunto de observaciones,
los elementos deben tener similitudes

Ejemplo:

    xxoopp         xx   oo    ppp
    oooppp         xx   ooo   pppp
    xxpp

> 1. Seleccionar un valor para K (centroides)
> 2. Asignamos cada uno de los elementos restantes al centro mas cercano
> 3. Asignamos cada punto a su centroide mas cercano
> 4. Repetimos paso 2 y 3 hasta que los centros no se modifiquen

#### _Metodo_del_codo_

* Calcula el agrupamiento para diferentes K
* El error al cuadrado para cada punto es el cuadrado de la distancia 
del punto desde su centro


## Links Utiles

* archive.ics.uci.edu/ml/index.php

* kaggle.com