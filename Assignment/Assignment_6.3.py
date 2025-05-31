'''
3. Use loan_data.csv dataset. Perform Naive Bayes classification to predict if the customer repays
the loan in full (not.fully.paid.column). Use all the other columns as features/predictors.
'''
# Import necessary libraries
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Assignment/loan_data.csv')

# Check if there are any missing values
# print(df.isnull().sum())

# Handle missing values (if any)
# df.fillna(df.mean(), inplace=True)

# Convert categorical variables into numerical variables

# Use LabelEncoder to encode the purpose column
label_encoder = LabelEncoder()
df['purpose'] = label_encoder.fit_transform(df['purpose'])
print("-----------------------------------------------")

#Select credit.policy
X = df[['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 
        'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 
        'delinq.2yrs', 'pub.rec']]

#Select not.fully.paid  
y = df['not.fully.paid']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))