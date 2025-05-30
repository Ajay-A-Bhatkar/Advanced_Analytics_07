# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/Assignment/social_media_usage.csv')
#Extract Daily_Minutes_Spent Column only for facebook
df = df[['Daily_Minutes_Spent','App']]
df_Insta = df[df['App']=='Instagram']
print(df_Insta.head())

#Calculation of Q1, Q3, IQR
Q1 = np.percentile(df_Insta['Daily_Minutes_Spent'], 25)
Q3 = np.percentile(df_Insta['Daily_Minutes_Spent'], 75)
IQR = Q3 - Q1

# Calculation of Lower and Upper bound
print("Inter Quantile Range (IQR) for Instagram: ", IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("Lower Bound : ",lower_bound)
print("Upper Bound : ",upper_bound)

#Identifying Outliers
outlier_indices_Insta = df_Insta[(df_Insta['Daily_Minutes_Spent'] < lower_bound) | (df_Insta['Daily_Minutes_Spent'] > upper_bound)].index

# df_FB['Outlier'] = np.where(df_FB['Daily_Minutes_Spent'] > upper_bound, 1, np.where(df_FB['Daily_Minutes_Spent'] < lower_bound, 1, 0))
print(df_Insta.loc[outlier_indices_Insta])

#Outlier Analysis for Facebook
sns.boxplot(x='App', y='Daily_Minutes_Spent', data=df_Insta)
plt.title('Box Plot for Instagram')
plt.show()