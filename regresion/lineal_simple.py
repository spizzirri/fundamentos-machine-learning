import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
Pasos:
 1. Captura  de datos.
 2. Separacion de datos, 80% entrenamiento 20% pruebas
 3. Entrenamiento
"""

dataset = pd.read_csv('../archivos/salarios.csv')

# El -1 hace referencia a excluir el ultimo elemento del arreglo
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

"""
Con el metodo train_test_spli vamos separar un conjunto de datos para entrenarlos. 
- test_size: le decimos que el 20% de los datos van a ser para testear.
- random_state: con 0 le indicamos que siempre use los mismos datos.
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""
Entrenando el modelo
"""
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

"""
Visualizamos los datos para ver el resultado del entrenamiento
"""
viz_train = plt
# Indicamos que muestro los datos que se usaron para entrenar
viz_train.scatter(X_train, Y_train, color='blue')
# Indicamos que muestre la prediccion
viz_train.plot(X_train, regressor.predict(X_train), color='black')
viz_train.title('Salario vs Experiencia')
viz_train.xlabel('Experiencia')
viz_train.ylabel('Salario')
viz_train.show()

"""
Visualizamos los datos para ver como se comporta con el set de datos de pruebas
"""

viz_train = plt
# Indicamos que muestro los datos que se dejaron para probar
viz_train.scatter(X_test, Y_test, color='red')
# Indicamos que muestre la prediccion
viz_train.plot(X_train, regressor.predict(X_train), color='black')
viz_train.title('Salario vs Experiencia')
viz_train.xlabel('Experiencia')
viz_train.ylabel('Salario')
viz_train.show()


"""
Analizamos el score para ver que tan cercana es la prediccion
"""
score = regressor.score(X_test, Y_test)
print("El % de acierto es del ", score*100, "%")
