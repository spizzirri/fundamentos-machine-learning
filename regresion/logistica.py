import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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