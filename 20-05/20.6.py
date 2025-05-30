import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')

plt.figure(figsize=(16, 12))

#1. Scatter Plot
plt.subplot(3, 2, 1)
plt.scatter(df['age'], df['chol'])
plt.title('Age vs Cholestrol')


#2. Histogram
plt.subplot(3, 2, 2)
plt.hist(df['trestbps'], bins=20)
plt.title('Resting Blood Pressure Distribution')

#3. Box Plot
plt.subplot(3, 2, 3)
plt.boxplot(df['thalach'])
plt.title('Max Hear Rate')

#4. Line Plot
plt.subplot(3, 2, 4)
df_sorted = df.sort_values(by='age')  #Sort by age for the line plot
plt.plot(df_sorted['age'], df_sorted['chol'], linestyle='--', color= 'b')
plt.title('Age vs Cholestrol Line')

#5. Bar Plot
plt.subplot(3, 2, 5)
df['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')

#6. Pie Chart
plt.subplot(3, 2, 6)
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=True)
plt.title('Heart DiseaseTarget Distribution')
plt.tight_layout()
plt.show()