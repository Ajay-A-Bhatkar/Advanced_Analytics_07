import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')

ages = df.groupby("age").median().reset_index()
print(ages.head())

sns.lineplot(x="age", y="chol", data=ages, linestyle=":", marker="o", linewidth=1.7 )
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Age vs Cholesterol")
plt.show()

## Various plots using Seaborn
# Set seaborn style
sns.set(style="whitegrid")
plt.figure(figsize=(18, 14))

#1. Scatter plot
plt.subplot(3, 3, 1)
sns.scatterplot(data=df, x="age", y="chol", hue='gender', color="blue", alpha=0.7)
plt.title('Age v\s Cholesterol')

#2. Histogram
plt.subplot(3, 3, 2)
sns.histplot(df['trestbps'], kde=True, color="red")
plt.title('Resting Blood Pressure')

#3. Boxplot
plt.subplot(3, 3, 3)
sns.boxplot(data=df, x='target', y='thalach', palette='Set2')
plt.title('Max Heart Rate by Target')

#4. Line Plot
plt.subplot(3, 3, 4)
sns.lineplot(data=df, x='age', y='chol', color='orange')
plt.title('Age vs Cholesterol Line')

#5. Bar Plot (Improved count plot for 'cp' feature)
plt.subplot(3, 3, 5)
sns.countplot(data=df, x='cp', palette='Set1', hue='target')
plt.title('Count of Chest Pain Types')

#6. Heatmap (Correlation Matrix)
plt.subplot(3, 3, 6)
relevant_columns = ['age', 'chol', 'thalach', 'trestbps', 'oldpeak', 'target']
correlation_matrix = df[relevant_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Selected Variables')

plt.tight_layout()
plt.show()