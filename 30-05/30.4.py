#Resampling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,  KFold, cross_val_score, cross_val_predict    
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load the dataset
df = pd.read_csv('Advanced_Analytics_07/Assignment/Titanic-Dataset.csv')

df = df[['Survived', 'Pclass', 'Age', 'Fare', 'Sex', 'SibSp']].dropna()

# Encoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# Bootstraping
print("\n-------------------Bootstraping Example --------------\n")
n_iterations = 100 # Number of bootstrap samples(i.e. how many times to repeat)
n_size =  int(len(X) * 0.80)  #Each bootstrap sample size(80% of original data)
bootstrap_scores=[] # To store accuracy of each bootstrap iteration

for i in range (n_iterations):
    # Resample with replacement
    # Each sample may include repeated row from the original dataset
    X_resample, y_resample = resample(X, y, n_samples=n_size, random_state=i)
    
    # Train/Test split

    X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.2, random_state=i)
    # Train the model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    bootstrap_scores.append(accuracy)

print(f" Average accuracy over {n_iterations} bootstrapped samples: {np.mean(bootstrap_scores):.3f}")

# K-fold cross-validation
print("\n-------------------K-fold Cross-validation Example --------------\n")

# Define the number of folds
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = LogisticRegression(max_iter=500)

#Perform K-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Accuracy Score for each fold: {cv_scores}")
print(f"Average accuracy over {kf.n_splits} folds: {np.mean(cv_scores):.3f}")
print("\n---------------------------------------------------------\n")

