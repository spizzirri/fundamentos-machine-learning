from sklearn import datasets, metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Cargo los datos
wines = datasets.load_wine()

# Datos para clasificar
X_wines = wines.data
columns=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
         'magnesium', 'total_phenols', 'flavanoids',
         'nonflavanoid_phenols', 'proanthocyanins',
         'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
         'proline']

x = pd.DataFrame(X_wines, columns=columns)

# Clasificacion
Y_wines = wines.target
y = pd.DataFrame(Y_wines, columns=['Target'])

print(x.head(5))
print(y.head(5))

# Graficamos solo dos atributos
plt.scatter(x['alcohol'], x['color_intensity'], c='blue')
plt.xlabel('Alcohol')
plt.ylabel('color_intensity')
plt.show()

# Indicamos que queremos 4 clusters
model = KMeans(n_clusters= 4, max_iter= 1000)
# Entrenamos
model.fit(x)
y_labels = model.labels_

# Hacemos las prediccion
y_kmeans = model.predict(x)

# Obtenemos la precision lograda
accuracy = metrics.adjusted_rand_score(Y_wines, y_kmeans)
print("Precision: ", accuracy)

# Graficamos nuevamente
plt.scatter(x['alcohol'], x['color_intensity'], c=y_kmeans)
plt.xlabel('Alcohol')
plt.ylabel('color_intensity')
plt.show()