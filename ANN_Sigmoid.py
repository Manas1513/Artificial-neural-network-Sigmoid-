import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.models as km
from keras.models import Sequential
from keras import activations, initializers, regularizers, constraints
from keras.layers import Dense, Activation


dataset=pd.read_csv("Churn_Modelling.csv")

dataset.head()

dataset.tail()

dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

dataset_new = pd.get_dummies(dataset, ['Geography', 'Gender'], drop_first=True)
dataset_new.head()

X = dataset_new.drop(columns=["Exited"])
y = dataset_new['Exited']

dataset["Exited"].value_counts()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state =123)

train_X.shape[1]

sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

model = Sequential()
model.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.summary()

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_X.shape

model.fit(train_X, train_y, batch_size = 10, epochs = 10)

prob_test=model.predict(test_X)  # Probability 
prob_test

from sklearn import metrics
y_test_pred=np.where(prob_test>0.5,1,0)
print(metrics.classification_report(test_y, y_test_pred))

