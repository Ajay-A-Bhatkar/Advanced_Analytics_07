'''
Assignment 7 (Decision Tree, PCA) - 30/05/2025
=============================================================
1. Do PCA on Titanic Dataset

2. Create a decision tree on titanic dataset

3. Do PCA on Diabetes Dataset and implement the components to create a decision tree.
'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt



# 1. Do PCA on Titanic Dataset
# Load the diabetes dataset
df = pd.read_csv('Advanced_Analytics_07\Assignment\Titanic-Dataset.csv')
# Drop rows with missing values
df.dropna()
print(df.shape)
survived = df['Survived'] #we will need this later

# Check for missing values
print(df.isnull().sum())

#Coloumns to retain
no_encoding_scaling_needed_coloumns = ['SibSp']
numerical_columns = ['Age', 'Fare', 'Parch',]
categorical_columns = ['Sex', 'Embarked', 'Pclass']

#No encoding needed coloumn
no_encoding_needed_coloumns = df[no_encoding_scaling_needed_coloumns]

# Handle missing values in numerical columns
numerical_data = df[numerical_columns]
imputer_sum = SimpleImputer(strategy='mean')
numerical_data = imputer_sum.fit_transform(numerical_data)
numerical_df = pd.DataFrame(numerical_data, columns=numerical_columns)

#Encode categorical data
categorical_data = df[categorical_columns]

# One hot encoding for categoricsl coloumns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(categorical_data)

# Convert the encoded array to a DataFrame
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the numerical and categorical DataFrames
combined_data = pd.concat([numerical_df, categorical_df], axis=1)
pd.set_option('display.max_columns', None)
print(combined_data.head())
pd.reset_option('display.max_columns')

#Standardize the data
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)


'''
===================================================================================================
'''
# Apply PCA
pca = PCA(7)
X_pca = pca.fit_transform(combined_data_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b')
plt.title('Cummulative Variance Explained by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cummulative Variance Explained')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
plt.legend()
plt.grid()
plt.show()


#Diplay Results
print("Explained Variance Ratio (for each PCA): ", explained_variance)
print("Cummulative Variance: ", cumulative_variance)

#Display Loadings (contribution of orighinal features to each principal component)
loadings = pd.DataFrame(pca.components_.T, # Transpose to get features as rows
                        columns = [f'PCA{i+1}' for i in range(pca.n_components_)],
                        index  = combined_data.columns
                        )

print("\nFeatures Loadings fro each Principal Component: \n", loadings.round(3)) 




'''
=====================================================================================================
'''








#Standard scalar outputs a numpy array so we convert it back to a dataframe
df = pd.DataFrame(combined_data_scaled, columns=combined_data.columns, index=combined_data.index)
print(df.shape)

#Seperate features and target
X = df.copy()
y = survived

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Combine predicted values with actual values for comparison
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPredicted vs Actual Values:\n", comparison_df.head(20))  # Display the first 20 rows of the comparison DataFrame

# Print Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))

