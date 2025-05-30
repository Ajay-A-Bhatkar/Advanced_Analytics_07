# Decision Tree

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Advanced_Analytics_07/30-05/Bank Customer Churn Prediction.csv')

# Drop rows with missing values
df.dropna()
print(df.shape)
churn = df['churn'] #we will need this later

#Coloumns to retain
no_encoding_scaling_needed_coloumns = ['credit_card', 'active_member']
numerical_coloumns = ['credit_score', 'age', 'tenure', 'balance', 'num_of_products', 'estimated_salary']
categorical_coloumns = ['country', 'gender']

#No encoding needed coloumn
no_encoding_needed_coloumns = df[no_encoding_scaling_needed_coloumns]

# Handle missing values for numerical columns
numreical_data = df[numerical_coloumns]
imputer_num = SimpleImputer(strategy='mean')
numreical_data = imputer_num.fit_transform(numreical_data)
numerical_df = pd.DataFrame(numreical_data, columns=numerical_coloumns)

#Encode categorical coloumns
categorical_data = df[categorical_coloumns]

# One-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(categorical_data)

# Convert the encoded array to a DataFrame
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_coloumns))

# Concatenate the numerical and categorical DataFrames
combined_data = pd.concat([numerical_df, categorical_df], axis=1)
pd.set_option('display.max_columns', None)
print(combined_data.head())
pd.reset_options('display.max_columns')

#Standardize the data
scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)

#Standard scalar outputs a numpy array so we convert it back to a dataframe
df = pd.DataFrame(combined_data_scaled, columns=combined_data.columns, index=combined_data.index)
print(df.shape)

#Seperate features and target
X = df.copy()
y = churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Combine predicted values with actual values for comparison
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPredicted vs Actual Values:\n", comparison_df.head(20))  # Display the first 20 rows of the comparison DataFrame

# Print Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))

#Save the tree as DOT file
with open('bank_tree.dot', 'w') as f:
    export_graphviz(model, out_file=f, feature_names=X.columns, filled=True)

#Google  'dot file editor' and upload this file to see the decision tree
