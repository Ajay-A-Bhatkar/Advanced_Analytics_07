'''
2. For the above, use Logistic Regression
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Assignment/Social_Network_Ads.csv')

# Check if there are any missing values
# print(df.isnull().sum())

# Use LabelEncoder to encode the Gender column
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
print(df.head())
print("----------------------------------------")

# Select Age and EstimatedSalary for scaling
standar_scalar = StandardScaler()
df[['Age', 'EstimatedSalary']] = standar_scalar.fit_transform(df[['Age', 'EstimatedSalary']])
print(df.head())

#Select target variable and label
X = df[['Age', 'EstimatedSalary', 'Gender']]
y = df['Purchased']

# Train - Test spplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
logistic_regression = LogisticRegression()

# Train the model using the training sets
logistic_regression.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = logistic_regression.predict(X_test)

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