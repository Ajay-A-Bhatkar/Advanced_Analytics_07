#Sampling and population using systematic probability distribution
import pandas as pd
df_2015 = pd.read_csv('22-05/marathon_results_2015.csv')
df_2016 = pd.read_csv('22-05/marathon_results_2016.csv')
df_2017 = pd.read_csv('22-05/marathon_results_2017.csv')

#Concatenate the datasets into single DataFrame
df = pd.concat([df_2015, df_2016, df_2017])

#Perform systematic sampling (picking every 7th record)
# ::7 means pick every 7th record, from start to end
# ,: means pick all columns
sampled_df = df.iloc[::7]
#Perform systematic sampling (picking every 7th record)
sampled_row_count = len(df.iloc[::7])

#Get the total number of rows in the original DataFrame
original_row_count = len(df)

#Print the first few rows of the original and sampled DataFrames
print("\nOriginal DataFrame:")
print(df.head())
print("\nSampled DataFrame:")
print(sampled_df.head())

#Find the total number of male and female athletes:
total_males = len(df[df['M/F'] == 'M'])
total_females = len(df[df['M/F'] == 'F'])

sample_size = int(0.1 * original_row_count)
sample_males = int(0.1 * total_males)
sample_females = int(0.1 * total_females)

#Generate proportionate sampls of male and female athletes
male_sample = df[df['M/F'] == 'M'].sample(n = sample_males, random_state=42)
female_sample = df[df['M/F'] == 'F'].sample(n = sample_females, random_state=42)

#Combine the male and female samples into a single DataFrame

sample_df =pd.concat([male_sample,female_sample])
sample_row_count= len(sample_df)
print(sample_row_count)

# print the first few rows of the original and sampled dataframe
print(df.head())

# print the first few rows of the sampled dataframe
print(sample_df.head())

#Print the number of rows in the original and sampled DataFrames
print(f"Number of rows in the original DataFrame: {original_row_count}")
print(f"Number of rows in the sampled DataFrame: {sampled_row_count}")


