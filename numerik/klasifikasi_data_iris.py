import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv('/Volumes/DATA/semester 3/numerik/IRIS.csv')
iris.head()

Species = iris['species'].value_counts().reset_index()
Species

plt.figure(figsize=(8,8))
plt.pie(Species['count'],labels=['Iris-setosa','Iris-versicolor','Iris-virginica'],autopct='%1.3f%%',explode=[0,0,0])
plt.legend(loc='upper left')
plt.show()

sns.FacetGrid(iris, hue ='species', height = 4).map(plt.scatter,"petal_length","sepal_width").add_legend()
plt.show()

iris.isnull().sum()

iris['species'].value_counts()

x = iris.drop('species', axis = 1)
y = iris['species'] 

scaler = StandardScaler().fit(x)
x_transform = scaler.transform(x)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_onehot = tf.keras.utils.to_categorical(y_encoded)
x_train, x_test, y_train, y_test = train_test_split(x_transform, y_onehot, test_size=0.10, random_state=101)

input_size = x_train.shape[1]
iris.shape

