import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz


df=pd.read_csv('Advanced_Analytics_07/Assignment/diabetes.csv')

print(df)
#Columns to retain
numerical_cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']

# #Handling missing values for numerical columns
numerical_data = df[numerical_cols]
imputer_num = SimpleImputer(strategy='mean')
numerical_data = imputer_num.fit_transform(numerical_data)
numerical_df = pd.DataFrame(numerical_data,columns = numerical_cols)

# # Stanadard the data
scaler = StandardScaler()
data_scaler = scaler.fit_transform(numerical_df)

# # Apply PCA
pca = PCA(5)
X_pca = pca.fit_transform(data_scaler)

# # Variance captured by each component

explained_variance= pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# # Plot cumulative variance
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+  1),cumulative_variance , markersize = 10, marker ='p', linestyle ='--', color ='b')
plt.title('Cumulative variance explained by PCA components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative variance explained')
plt.axhline(y =0.9, color ='r', linestyle ='--', label ='90% Variance')
plt.axhline(y =0.95, color ='g', linestyle ='--', label ='95% Variance')
plt.legend()
plt.grid()
plt.show()


# # Display the results

print('Expalined Variance Ratio (for each PCA):')
print(explained_variance)
print('Cumulative Variance')
print(cumulative_variance)

# #Display loadings (contribution of original features to each principal component)

loadings= pd.DataFrame(
    pca.components_.T,    # Transpose to get features as row
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=numerical_cols
)

print("\n Feature Loading for each principal component")
print(loadings.round(3))    # Rounded for better readability



print('------------------------------------------DECISION TREE--------------------------------------------------------')
# Drop rows with missing values
df = df.dropna()
print(df.shape)
churn = df['Outcome']

# Columns to retain

numerical_cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']
#categorical_cols = ['gender']


# Handle missing values for numerical columns
numerical_data = df[numerical_cols]
imputer_num= SimpleImputer(strategy = 'mean')
numerical_data = imputer_num.fit_transform(numerical_data)
numerical_df = pd.DataFrame(numerical_data, columns= numerical_cols)

# # Encode categorical columns
# categorical_data = df[categorical_cols]
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# categorical_data= encoder.fit_transform(categorical_data)
# categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_cols))

# Combine numerical and categorical data using pd.concat
combined_data= pd.concat([ numerical_df],axis =1)
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
with open("diabetes.dot", "w") as f:
  export_graphviz(model, out_file=f, feature_names = combined_data.columns , filled= True)

# Google 'dot file editor' and upload this file to see the decision tree

