import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Analysis"""

diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.min()

diabetes_dataset.max()

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 --> Non-Diabetic

1 --> Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Data Standardization"""

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

"""Model Evaluation --- Accuracy Score"""

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)


import pickle

# Save the model to disk
filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

print("Model saved successfully!")
