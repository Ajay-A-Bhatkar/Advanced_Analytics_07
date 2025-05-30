#Doing PCA on the diabetes dataset
#PCA : Principal Component Analysis

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the diabetes dataset
df = pd.read_csv('Advanced_Analytics_07/Notes/diabetes.csv')

# Coloumns to retain
numerical_coloumns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

#Handle missing values for numeric columns
numerical_data = df[numerical_coloumns]
imputer_num = SimpleImputer(strategy='mean')
numerical_data = imputer_num.fit_transform(numerical_data)
numerical_df = pd.DataFrame(numerical_data, columns=numerical_coloumns)

#Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(numerical_df)

# Perform PCA
pca = PCA(n_components=3)  # Specify the number of components you want to retain
X_pca = pca.fit_transform(data_scaled)

#Variance captured by each principal component
explained_variance = pca.explained_variance_ratio_
cummulative_variance = np.cumsum(explained_variance)

# Plot the explained variance
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
print("Cummulative Variance: ", cummulative_variance)

#Display Loadings (contribution of orighinal features to each principal component)
loadings = pd.DataFrame(pca.components_.T, # Transpose to get features as rows
                        columns = [f'PCA{i+1}' for i in range(pca.n_components_)],
                        index=numerical_coloumns
                        )

print("\nFeatures Loadings fro each Principal Component: \n", loadings.round(3))  #Rounded for better readability