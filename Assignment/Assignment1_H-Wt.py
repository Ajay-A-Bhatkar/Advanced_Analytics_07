'''
2)Use Dataset weight-height.csv. Do Outlier Analysis for :
1. Female Heights
2. Female weights
3. Male Heights
4. Male weights
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/Assignment/weight-height.csv")

# Function to perform outlier analysis
def outlier_analysis(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    return outliers


# 1. Female Heights
df_female = df[['Gender','Height']]
female_heights = df_female[df_female['Gender'] == 'Female']
outliers_female_heights = outlier_analysis(female_heights, 'Height')
print("Outliers in Female Heights:")
print("******************************************************************************")
print("Number of outliers in Female Heights:", len(outliers_female_heights))
print("******************************************************************************")

# 2. Female weights
df_female = df[['Gender','Weight']]
female_weights = df_female[df_female['Gender'] == 'Female']
outliers_female_weights = outlier_analysis(female_weights, 'Weight')
print("Outliers in Female weights:")
print("******************************************************************************")
print("Number of outliers in Female weights:", len(outliers_female_weights))
print("******************************************************************************")

# 3. Male Heights
df_male = df[['Gender','Height']]
male_heights = df_male[df_male['Gender'] == 'Male']
outliers_male_heights = outlier_analysis(male_heights, 'Height')
print("Outliers in Male Heights:")
print("******************************************************************************")
print("Number of outliers in Male Heights:", len(outliers_male_heights))
print("******************************************************************************")

# 4. Male weights
df_male = df[['Gender','Weight']]
male_weights = df_male[df_male['Gender'] == 'Male']
outliers_male_weights = outlier_analysis(male_weights, 'Weight')
print("Outliers in Male weights:")
print("******************************************************************************")
print("Number of outliers in Male weights:", len(outliers_male_weights))
print("******************************************************************************")

