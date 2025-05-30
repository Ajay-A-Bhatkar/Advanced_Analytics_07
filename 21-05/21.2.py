#Outlier Analysis using IQR method and boxplot
#IQR = Q3 - Q1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/21-05/temp.csv')

#Extract NY Column only
ny_data = df['NY']

#Calculation of Q1, Q3, IQR
Q1 = np.percentile(ny_data, 25)
Q3 = np.percentile(ny_data, 75)
IQR = Q3 - Q1
# print(IQR)

#Set lower and upper bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#Identify outlier indices
#True/False applied to the original dataframe, then .index will give us the row number for the matching condition in the dataframe
outlier_indices = ny_data[(ny_data < lower_bound) | (ny_data > upper_bound)].index
print(outlier_indices) #Row numbers of the outlier records

#Print Outlier Rows
print("Outliers in NY Temperatures using IQR Method:")
print(df.loc[outlier_indices]) #Apply the row nos. to the original dataframe

#Show Boxplot for NY Temperatures
plt.figure(figsize=(10, 6))
sns.boxplot(y=ny_data)
plt.title('Boxplot for NY Temperatures')
plt.show()