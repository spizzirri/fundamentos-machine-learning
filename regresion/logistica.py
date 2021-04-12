import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

diabetes = pd.read_csv('../archivos/diabetes.csv')

# Todo los valores menos la ultima columna "Outcome"
feature_cols = diabetes.columns.values[:-1]
print("Columnas del archivo: \n", feature_cols)

x = diabetes[feature_cols]
# Nuestro valor Y de referencia es la columna Outcome
y = diabetes.Outcome

"""
Separamos los datos
"""
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

"""
Entrenamos el modelo
"""
logreg = LogisticRegression(max_iter=len(X_train))
logreg.fit(X_train, y_train)

"""
Vemos la prediccion
"""
y_pred = logreg.predict(X_test)

print("\nPrediccion\n")
print(y_pred)

"""
Generando matriz de confusion con:
y_test: la informacion que nosotros brindamos.
y_pred: la informacion que predice nuestro modelo
"""
cnf_matriz = metrics.confusion_matrix(y_test, y_pred)

"""
Pedimos que nos muestre dos valores:
0 -> sino tiene diabetes
1 -> si tiene diabetes
""" 
class_names = [0, 1]
fig, ax = plt.subplots()
tick_mark = np.arange(len(class_names))
plt.xticks(tick_mark, class_names)
plt.yticks(tick_mark, class_names)

sns.heatmap(pd.DataFrame(cnf_matriz), annot = True, cmap='Blues_r', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Matriz de confusion', y = 1.1)
plt.ylabel('Etiqueta actual')
plt.xlabel('Etiqueta de prediccion')
plt.show()

print("Exactitud: ", metrics.accuracy_score(y_test, y_pred))