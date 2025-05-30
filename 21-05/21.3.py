#Covariance and Correlation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/Assignment/weight-height.csv')

print(df.head())

#separate the data based on genders
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

#Calculate corelation 
corr_overall = df[['Height','Weight']].corr()
corr_male = df_male[['Height','Weight']].corr()
corr_female = df_female[['Height','Weight']].corr()
#Calculate covariance
cov_overall = df[['Height','Weight']].cov()
cov_male = df_male[['Height','Weight']].cov()
cov_female = df_female[['Height','Weight']].cov()

print("Overall Correlation between Height and weight: ")
print(corr_overall)

print("Male Correlation between Height and weight: ")
print(corr_male)

print("Female Correlation between Height and weight: ")
print(corr_female)

print("\n Overall Covariance between Height and weight: ")
print(cov_overall)

print("\n Male Covariance between Height and weight: ")
print(cov_male)

print("\n Female Covariance between Height and weight: ")
print(cov_female)

#Create scatter plots with correlation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#Overall data plot
axes[0].scatter(df['Height'], df['Weight'], color='blue', alpha=0.5)
axes[0].set_title(f'Overall Correlation: {corr_overall.loc['Height','Weight']:.2f}')
axes[0].set_xlabel('Height')
axes[0].set_ylabel('Weight')
axes[0].grid(True)

#Male data plot
axes[1].scatter(df_male['Height'], df_male['Weight'], color='green', alpha=0.5)
axes[1].set_title(f'Male Correlation: {corr_male.loc["Height","Weight"]:.2f}')
axes[1].set_xlabel('Height')
axes[1].set_ylabel('Weight')
axes[1].grid(True)

#Female data plot
axes[2].scatter(df_female['Height'], df_female['Weight'], color='pink', alpha=0.5)
axes[2].set_title(f'Female Correlation: {corr_female.loc["Height","Weight"]:.2f}')
axes[2].set_xlabel('Height')
axes[2].set_ylabel('Weight')
axes[2].grid(True)

plt.tight_layout()
plt.show()


#Heatmap
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(corr_overall, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Overall Correlation Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
