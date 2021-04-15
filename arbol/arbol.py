import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#sns.set()

test_df = pd.read_csv('../archivos/titanic-test.csv')
train_df = pd.read_csv('../archivos/titanic-train.csv')

print(train_df.head(5))
print(train_df.info())

print("Mostramos un grafico de barras de personas sobrevivientes separados entre M y F")
train_df.Sex.value_counts().plot(kind = 'bar', color=['b', 'r'])
plt.title('Distribucion de sobrevivientes')
plt.show()

print("Primeras 5 filas antes de acomodar los datos para entrenar")
print(train_df.head())

print("Preparacion de los datos")
label_encoder = preprocessing.LabelEncoder()
# Transformamos los valores male y female en 0 y 1
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])

# Las filas de la columna Age, que no esten cargados, se les pondra el valor medio
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Las filas de la columna Embarked, que no esten cargados, se les pondra el valor S
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# Eliminamos los datos de las siguientes columnas.
train_predictors = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

# Evaluamos las columnas de tipo objeto que nos quedaron luego del drop.
categorical_cols = [cname for cname in train_predictors.columns if
                        train_predictors[cname].nunique() < 10 and
                        train_predictors[cname].dtype == 'object'
                    ]

# Evaluamos las columnas con valores numericos que nos quedaron luego del drop
numerical_cols = [cname for cname in train_predictors.columns if
                    train_predictors[cname].dtype in ['int64', 'float64']
                 ]

# Unimos nuestras nuevas columnas
my_cols = categorical_cols + numerical_cols

print("Columnas con las que nos quedamos para entrenar")
print(my_cols)

print("Actualizamos los datos con los que se usaran para predecir")
train_predictors = train_predictors[my_cols]

# Generamos informacion dummy para probar
dummy_encoded_train_predictors = pd.get_dummies(train_predictors)

print("train_predictors")
print(train_predictors.head(3))

print("Informacion dummy")
print(dummy_encoded_train_predictors.head(3))

print("Diversidad de valores columnas Pclass")
print(train_df['Pclass'].value_counts())

#Valores que queremos lograr predecir
y_target = train_df['Survived'].values

#Set de datos ya preparados
X_features_one = dummy_encoded_train_predictors.values

# Con random_state = 1 se indica que cada vez que corra el algoritmo se tome una seccion distinta
x_train, x_validation, y_train, y_validation = train_test_split(X_features_one, y_target, test_size = 0.25, random_state=1)

tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(X_features_one, y_target)

tree_one_accuracy = round(tree_one.score(X_features_one, y_target), 4)
print('Accuracy: %0.4f' %(tree_one_accuracy))