#Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
df = pd.read_csv('Assignment/diabetes.csv')

#Split into predictorvariables X and target variable y
X =  df.drop('Outcome', axis=1)
y = df['Outcome']

#  Standardize only the insulin coloumn
scaler = StandardScaler()
X[['Insulin']] = scaler.fit_transform(X[['Insulin']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Create and train a logistic regression model
#Options : liblinear : for small datasets, binary classification, can work w/o standardization
#lbfgs : Multi class, medium - to - large datasets, good accuracy, newton-cg :similar
#sag : for large datasets, good accuracy, needs standardization

classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)

#Make predictionS
y_test_prediction = classifier.predict(X_test)
y_train_prediction = classifier.predict(X_train)

#Display predictions comparison
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_prediction})
print(comparison.head(10))

#Print Accuracy score
print('Accuracy score:', accuracy_score(y_test, y_test_prediction))
print('Training accuracy score:', accuracy_score(y_train, y_train_prediction))
print(classification_report(y_test, y_test_prediction))


#H0 : The patient does not have diabetes (Outcome = 0)
#H1 : The patient has diabetes (Outcome = 1)
#FP (Type I Error): The model predicts the patient has diabetes, but in reality, they do not have diabetes.
#FN (Type II Error): The model predicts the patient does not have diabetes, but in reality, they do have diabetes.

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_test_prediction)
print("\nConfusion Matrix:")
print(conf_mat)

# Sequence R 1 C 1: TN,      R 1 C 2: FP,    R 2 C 1: FN,    R 2 C 2: TP

#Visualize the confusion matrix using heatmap
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_prediction))

#Extract TP, TN, FP, FN from the confusion matrix
conf_mat = confusion_matrix(y_test, y_test_prediction)

TN = conf_mat[0,0]
FP = conf_mat[0,1]
TP = conf_mat[1,1]
FN = conf_mat[1,0]

print("\nTrue Positives (TP):", TP)
print("\nTrue Negatives (TN):", TN)
print("\nFalse Positives (FP):", FP)
print("\nFalse Negatives (FN):", FN)

#Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_test_prediction)
prediction = precision_score(y_test, y_test_prediction)
recall = recall_score(y_test, y_test_prediction)
f1 = f1_score(y_test, y_test_prediction)

#Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {prediction:.2f}")   
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


