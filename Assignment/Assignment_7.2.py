'''
Assignment 7 (Decision Tree, PCA) - 30/05/2025
=============================================================
1. Do PCA on Titanic Dataset

2. Create a decision tree on titanic dataset

3. Do PCA on Diabetes Dataset and implement the components to create a decision tree.
'''
# 2. Create a decision tree on titanic dataset

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/content/titanic.csv")

# Drop rows with missing values
df = df.dropna()
print(df.shape)
churn = df['survived']  # need this later

df=df.drop(columns='embarked')
df=df.drop(columns='deck')


# Columns to retain

numerical_cols = ['pclass','age','sibsp','parch','fare']
categorical_cols = ['gender']


# Handle missing values for numerical columns
numerical_data = df[numerical_cols]
imputer_num= SimpleImputer(strategy = 'mean')
numerical_data = imputer_num.fit_transform(numerical_data)
numerical_df = pd.DataFrame(numerical_data, columns= numerical_cols)

# Encode categorical columns
categorical_data = df[categorical_cols]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_data= encoder.fit_transform(categorical_data)
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_cols))

# Combine numerical and categorical data using pd.concat
combined_data= pd.concat([ numerical_df,categorical_df],axis =1)
pd.set_option('display.max_columns', None)
print(combined_data.head())
pd.reset_option("display.max_columns")

# Standardize the data
scaler= StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)

# Stnadard scaler outputs a numpy array so we need to convert it into a adataframe
df = pd.DataFrame(combined_data_scaled, columns= combined_data.columns, index = combined_data.index)
print(df.shape)

# Separate features and get target
X = df.copy()
y = churn

X_train, X_test, y_train, y_test= train_test_split(X, y , test_size = 0.2, random_state= 1)

model = DecisionTreeClassifier()
model= model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Combined predicted and actual values for comparison
comparison_df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print("\n Predicted v/s Actual values:")
print(comparison_df.head(20))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save Tre as DOT file
with open("bank_tree.dot", "w") as f:
  export_graphviz(model, out_file=f, feature_names = combined_data.columns , filled= True)

# Google 'dot file editor' and upload this file to see the decision tree