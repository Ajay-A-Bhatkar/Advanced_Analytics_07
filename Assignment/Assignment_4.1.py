'''
Assignment: For the three combined Marathon datasets, extract the top 25 records with lowest finish time (Official Time column). 
Check the null hypothesis HO = The average finishing time of the top 25 athletes is 130 minutes.
Test it for a = 1%, 5%, and 10% using Two tailed test
'''
import pandas as pd
from scipy import stats
from scipy.stats import norm
import numpy as np

df_2015 = pd.read_csv('Assignment/marathon_results_2015.csv')
df_2016 = pd.read_csv('Assignment/marathon_results_2016.csv')
df_2017 = pd.read_csv('Assignment/marathon_results_2017.csv')

df_combined = pd.concat([df_2015, df_2016, df_2017])



# Split the 'Official Time' column into hours, minutes, and seconds
df_combined[['Hours', 'Minutes', 'Seconds']] = df_combined['Official Time'].str.split(':', expand=True)

# Convert hours, minutes, and seconds to integers
df_combined['Hours'] = df_combined['Hours'].astype(int)
df_combined['Minutes'] = df_combined['Minutes'].astype(int)
df_combined['Seconds'] = df_combined['Seconds'].astype(int)

# Calculate total time in seconds
df_combined['Official Time (seconds)'] = df_combined['Hours'] * 3600 + df_combined['Minutes'] * 60 + df_combined['Seconds']

# Sort the DataFrame by 'Official Time (seconds)' in ascending order
df_combined = df_combined.sort_values('Official Time (seconds)')
df_combined = df_combined.head(25)
print(df_combined)

print("=====================================================================")

# Pick top 25 records : df = df.iloc[:25]
df_top25 = df_combined.iloc[:25]
print(df_top25)

print("=====================================================================")


mu0 = 7933.2

#Sample stats
mean = df_top25['Official Time (seconds)'].mean()
print(mean)
std = df_top25['Official Time (seconds)'].std()
se = std / np.sqrt(len(df_top25))

#Z-Test (one-tailed)
z_stat = (mean - mu0) / se
p_value = norm.cdf(z_stat)

print("=====================================================================")

alphas =[0.01, 0.05, 0.10]
print(f"{'Alpha':<8}{'Z-Stat':>10}{'P-Value':>12}{'Critical Z':>15}{'Decision':>15}")
print("-" * 60)

for alpha in alphas:
    critical_z = norm.ppf(1 - alpha) # one tailed critical value
    descision = "Reject H0" if (z_stat) > critical_z else "  Fail to Reject H0"
    print(f"{alpha:<8.2f}{z_stat:10.4f}{p_value:12.4f}{critical_z:15.4f}{descision:>15}")