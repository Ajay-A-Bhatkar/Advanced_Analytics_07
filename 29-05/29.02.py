# Naive Bayes classifier - GaussianNaiveBayes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Load the dataset
df = pd.read_csv('29-05/diabetes.csv')

#Define coloums for scaling
standar_scalar_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction', 'Age']
minmax_scalar_cols = ['SkinThickness','Insulin', 'BMI']

# Scaling the data using StandardScaler
standard_scaler = StandardScaler()
df[standar_scalar_cols] = standard_scaler.fit_transform(df[standar_scalar_cols])

# Scaling the data using MinMaxScaler
minmax_scaler = MinMaxScaler()
df[minmax_scalar_cols] = minmax_scaler.fit_transform(df[minmax_scalar_cols])

# Display the coloumns temporily
pd.set_option('display.max_columns', None)
print(df.head())
pd.reset_option('display.max_columns', None)

# Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
gaussian_classifier = GaussianNB()

# Train the classifier on the training data
gaussian_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gaussian_classifier.predict(X_test)

# Evaluate the classifier

accuracy = accuracy_score(y_test, y_pred)
print("----------------------------------------------------------------------------------")
print("Accuracy:", accuracy)
print("----------------------------------------------------------------------------------")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("----------------------------------------------------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("----------------------------------------------------------------------------------")
print("Recall:", recall_score(y_test, y_pred))
print("----------------------------------------------------------------------------------")
print("Precision:", precision_score(y_test, y_pred))
print("----------------------------------------------------------------------------------")
print("F1 Score:", f1_score(y_test, y_pred))
print("----------------------------------------------------------------------------------")

print("Prdicted Vs Actual Outcomes (Test sets):")
for actual, predicted in zip(y_test, y_pred):
    print(f" Actual: {actual}, Predicted: {predicted}")

