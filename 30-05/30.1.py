import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset

df = pd.read_csv('30-05/pca_student_test_scores.csv')

#Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

#Plot spelling Vs vocabulary
axs[0, 0].scatter(df['spelling'], df['vocabulary'], color='blue', alpha=0.5)
axs[0, 0].set_title('spelling Vs vocabulary')
axs[0, 0].set_xlabel('spelling')
axs[0, 0].set_ylabel('vocabulary')

# Plot vocabulary Vs multiplication
axs[0, 1].scatter(df['vocabulary'], df['multiplication'], color='green', alpha=0.5)
axs[0, 1].set_title('vocabulary Vs multiplication')
axs[0, 1].set_xlabel('vocabulary')
axs[0, 1].set_ylabel('multiplication')

#Plot spelling Vs multiplication
axs[1, 0].scatter(df['spelling'], df['multiplication'], color='red', alpha=0.5)
axs[1, 0].set_title('spelling Vs multiplication')
axs[1, 0].set_xlabel('spelling')
axs[1, 0].set_ylabel('multiplication')

# Multiplication Vs Geometry
axs[1, 1].scatter(df['multiplication'], df['geometry'], color='purple', alpha=0.5)
axs[1, 1].set_title('multiplication Vs geometry')
axs[1, 1].set_xlabel('multiplication')
axs[1, 1].set_ylabel('geometry')

#Adjust the layout of the subplots
plt.tight_layout()
plt.show()

# Calculate the correlation matrix
corr_matrix = df.corr()
print("Correlation Matrix:")
print(corr_matrix)

#Plot the heatmapp for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()