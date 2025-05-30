'''
1. Use dataset Retail_Data_Transactions.csv. Find average daily sale count. 
Use pisson distribution to calculate and plot probability for average daily sale count: +/- 10 transactions.
'''
# Import necessary libraries
from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
# Load the data
df = pd.read_csv('Assignment/Retail_Data_Transactions.csv')
# Calculate average daily sale count
df['Date'] = pd.to_datetime(df['trans_date'])
# df['Day'] = df['Date'].dt.day
# df['Month'] = df['Date'].dt.month
# df['Year'] = df['Date'].dt.year
# df['Day_of_Week'] = df['Date'].dt.dayofweek
# print(df.head())

df_new = df.groupby('Date')['trans_date'].count()
print(df_new)
# avg_daily_sale = count.mean() 


# Calculate average daily sale count
avg_daily_sale = df_new.mean()
print(f"\nAverage daily sale count: {avg_daily_sale:.2f}")

# Define the range for which to calculate the probability
lower_bound = int(avg_daily_sale - 10)
upper_bound = int(avg_daily_sale + 10)

# Ensure bounds are non-negative
if lower_bound < 0:
    lower_bound = 0


# Calculate the probability for each value in the range using Poisson distribution
probabilities = [poisson.pmf(k, avg_daily_sale) for k in range(lower_bound, upper_bound + 1)]

# Plot the probability distribution
plt.figure(figsize=(10, 6))
plt.bar(range(lower_bound, upper_bound + 1), probabilities)
plt.xlabel('Daily Sale Count')
plt.ylabel('Probability')
plt.title(f'Poisson Probability Distribution for Average Daily Sale Count (+/- 10)')
plt.xticks(range(lower_bound, upper_bound + 1))
plt.grid(axis='y', linestyle='--')
plt.show()

# Calculate the cumulative probability for the given range
cumulative_probability = sum(probabilities)
print(f"Probability of daily sale count being between {lower_bound} and {upper_bound}: {cumulative_probability:.4f}")




