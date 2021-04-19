"""
AGRUPANDO LOS DATOS

* Virginica, Versicolor y Setosa
* 50 muestras de cada especie
* Largo y ancho del sepalo

"""

from sklearn.cluster import KMeans
from sklearn import datasets, metrics
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X_iris = iris.data
Y_iris = iris.target

x = pd.DataFrame(iris.data, columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns = ['Target'])

print("Data:\n", x.head(5))
print("\nTarget:\n", y.head(5))

plt.scatter(x['Petal Length'], x['Petal Width'], c='blue')
plt.xlabel('Petal Length', fontsize=10)
plt.ylabel('Petal Width', fontsize=10)
plt.show()


model = KMeans(n_clusters= 3, max_iter= 1000)
model.fit(x)
y_labels = model.labels_

y_kmeans = model.predict(x)
print("Predicciones ", y_kmeans)

accuracy = metrics.adjusted_rand_score(Y_iris, y_kmeans)
print("Precision: ", accuracy)

plt.scatter(x['Petal Length'], x['Petal Width'], c=y_kmeans, s=30)
plt.xlabel('Petal Length', fontsize = 10)
plt.ylabel('Petal Width', fontsize=10)
plt.show()