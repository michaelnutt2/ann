# -*- coding: utf-8 -*-
"""
In this assignment:
·        You will learn how to use KERAS https://www.tensorflow.org/guide/keras/sequential_model
·        You learn how build a simple Sequential Feed forward Neural Network
.        You will learn how to split your data into training tuning and test
.        You will learn how to use the TUNING dataset to tune the hyperparameters of your model

Dataset
In this assignment, you will apply an ANN to a binary classification dataset, the Pima Indians diabetes dataset.
We will try to build an ANN to determine if a person has diabetes or not

The dataset has the following 8 attributes, the last is a binary value of 0 or 1.
0 = tested negative for diabetes
1= tested positive for diabetes

Attributes

   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

"""

import subprocess
import sys


def install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("keras")
install("pandas")
install("sklearn")
install("numpy")

# import libraries
# first neural network with keras tutorial
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# load the dataset
dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
                  delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# INSTRUCTIONS
# 1.  Split your dataset into training and test set using the sklearn.model_selection.train_test_split.
# Let the training set have 70% of the data and the test set have 30% of the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# 2.  Split your dataset into training dataset into training and tuning dataset using the sklearn.model_selection.train_test_split.
# Let the training set have 80% of the data and the tuning dataset have 20% of the data.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.2, random_state=42)

# Set the model to be a simple feed-forward (layered) architecture
# See https://keras.io/api/models/ and https://keras.io/api/models/sequential/
# not to be confused with a sequence-based alg/model to process sequential data
model = keras.Sequential()
# Add a fully connected (dense) hidden layer with 12 relu activation units
# See https://keras.io/api/models/sequential/, set input_dim=8
model.add(Dense(12, input_dim=8, activation='relu'))

# Add a fully connected (dense) hidden layer with 8 relu activation units
# See https://keras.io/api/models/sequential/
model.add(Dense(8, activation='relu'))

# Add a dense output layer with a single sigmoid activation unit
model.add(Dense(1, activation='sigmoid'))

# Compile the model, specifying (1) the Adam optimizer,
# (2) the 'BinaryCrossentropy' loss function, and (3) metrics=['accuracy']
# See https://keras.io/api/models/model_training_apis/

model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the _TRAIN_ dataset set epochs to 100 and batch_size to 32
model.fit(X_train, y_train, epochs=100, batch_size=64)

# evaluate the keras model with TUNING DATASET 
_, accuracy = model.evaluate(X_valid, y_valid)
print('Accuracy: %.2f' % (accuracy * 100))

# NOW, increase/reduce the number of epochs, batch size, number of hidden units, number of hidden layers... and see
# what values give you the best results on the TUNING dataset DO NOT TOUCH THE TEST DATASET YET UNTIL YOU ARE DONE
# MAKING THESE DECISIONS.

# FINALLY, evaluate your model on TEST dataset
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
