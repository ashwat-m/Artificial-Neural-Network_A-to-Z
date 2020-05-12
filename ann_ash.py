# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:45:42 2020

@author: Ashwat Mahendran
"""
#Part 1 - Data Preprocessing
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Encoding Categorical Data (Independent Variable)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])
ct = ColumnTransformer([("1", OneHotEncoder(), [1])], remainder = 'passthrough')
X=ct.fit_transform(X)
X = X[:, 1:]

#Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part 2 - Building ANN
#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(activation='relu', input_dim = 11, units = 6, kernel_initializer = "uniform"))
#classifier.add(Dropout(rate = 0.1))

#Adding the second hidden layer
classifier.add(Dense(activation='relu', units = 6, kernel_initializer = "uniform"))

#Adding the output layer
classifier.add(Dense(activation='sigmoid', units = 1, kernel_initializer = "uniform"))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training Set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)


#Part 3 - Making predictions and Evaluating the Model
#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Predicting a single new observation
#new_pred = classifier.predict(sc.transform(np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])))
#new_pred = (new_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


#Part 4 - Evaluating, Improving and Tuning the ANN
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim = 11, units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation='relu', units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation='sigmoid', units = 1, kernel_initializer = "uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()    

#Improving the ANN
#Dropout Regularization to reduce overfitting if needed
#Refer Above

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim = 11, units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation='relu', units = 6, kernel_initializer = "uniform"))
    classifier.add(Dense(activation='sigmoid', units = 1, kernel_initializer = "uniform"))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size' : [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, Y_train)

best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_









