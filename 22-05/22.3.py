#Cluster Sampling
import pandas as pd
import numpy as np

df_2015 = pd.read_csv('22-05/marathon_results_2015.csv')
df_2016 = pd.read_csv('22-05/marathon_results_2016.csv')
df_2017 = pd.read_csv('22-05/marathon_results_2017.csv')

#Concatenate the datasets into single DataFrame
df = pd.concat([df_2015, df_2016, df_2017])

#Get the total number of rows in the original DataFrame
original_row_count = len(df)

#Find the unique countries the athletes are from
unique_countries = df['Country'].unique()
num_countries = len(unique_countries)

#Randomly select half of the countries
np.random.seed(42)

#choice() function in numpy generates a random sample from an array
selected_countries = np.random.choice(unique_countries, size=num_countries//2, replace=False) #replace=False  means donot allow same country twice or more

#Create a new DataFrame containing all records of the selected countries
sample_df = df[df['Country'].isin(selected_countries)]

#Get the total number of rows in the sample DataFrame
sample_row_count = len(sample_df)

#Print the first few rows of the original and sample DataFrames
print("First few rows of the Original DataFrame:")
print(df.head())
print("First few rows of the Sample DataFrame:")
print(sample_df.head())

#Print the number of rows in the original and sample DataFrames
print(f"\nNumber of rows in the Original DataFrame: {original_row_count}")
print(f"Number of rows in the Sample DataFrame: {sample_row_count}")