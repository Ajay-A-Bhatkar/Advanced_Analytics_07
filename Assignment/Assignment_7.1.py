'''
Assignment 7 (Decision Tree, PCA) - 30/05/2025
=============================================================
1. Do PCA on Titanic Dataset

2. Create a decision tree on titanic dataset

3. Do PCA on Diabetes Dataset and implement the components to create a decision tree.
'''
# 1. Do PCA on Titanic Dataset
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('Advanced_Analytics_07/Assignment/Titanic-Dataset.csv')

df=df.drop(columns='Embarked')



#Encoding
df= pd.get_dummies(df,columns=['Sex'])


print(df)
#Columns to retain
numerical_cols = ['Pclass','Age','Sex_male','Sex_female','SibSp','Fare','Parch']

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
plt.plot(range(1, len(cumulative_variance)+  1),cumulative_variance , marker ='o', linestyle ='--', color ='b')
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



