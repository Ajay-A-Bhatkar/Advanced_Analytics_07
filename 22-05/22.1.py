#Sampling and population using simple probability distribution
import pandas as pd

df_2015 = pd.read_csv('22-05/marathon_results_2015.csv')
df_2016 = pd.read_csv('22-05/marathon_results_2016.csv')
df_2017 = pd.read_csv('22-05/marathon_results_2017.csv')

#Concatenate the datasets into single DataFrame
df = pd.concat([df_2015, df_2016, df_2017])

#Get the total number of rows in the original DataFrame
original_row_count = len(df)

#Perform simple random sampling (selecting 1/10th of the records)
sampled_df = df.sample(frac=0.1, random_state=42)

#Get the total number of rows in the sampled DataFrame
sampled_row_count = len(sampled_df)

print(df.head())
print(sampled_df.head())

#Print the number of rows in the original and sampled DataFrames
print(f"Number of rows in the original DataFrame: {original_row_count}")
print(f"Number of rows in the sampled DataFrame: {sampled_row_count}")


