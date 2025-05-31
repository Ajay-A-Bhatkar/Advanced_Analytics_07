# Exponential Distribution

import pandas as pd
import numpy as np
from math import exp
import matplotlib.pyplot as plt

df = pd.read_csv('Advanced_Analytics_07/31-05/orders.csv')

# Combine 'date and 'timer' into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True)

#Sort by datetime
df.sort_values(by='datetime')

#Group by date and calculate intra-day interarrival times
df['date_only'] = df['datetime'].dt.date # To avoid overwritting 'date' column if needed

time_diffs = []

for _, group in df.groupby('date_only'):
    group = group.sort_values(by='datetime')
    intra_day_diffs = group['datetime'].diff().dt.total_seconds().dropna() / 60  # Convert to minutes
    time_diffs.extend(intra_day_diffs.tolist())

# Convert to numpy array
time_diffs = np.array(time_diffs)

#Expotemtial distribution parameters
mean_time_diff = time_diffs.mean()
rate_Lambda = 1 / mean_time_diff

#Probability that next customer arrives within 3 minutes
t = 3  # minutes
probability_after_3_min = 1 - exp(-rate_Lambda * t)

#Output results
print(f"Average time between customers (intra-day): {mean_time_diff:.2f} minutes")
print(f"λ (rate): {rate_Lambda:.4f}")
print(f"Probability that the next customer arrives AFTER {t} minutes: {probability_after_3_min:.4f}")

#Calculate and plot the PDF of the exponential distribution
x = np.linspace(0, 19, 20)  # 20 minutes range
pdf = rate_Lambda * np.exp(-rate_Lambda * x)
cdf = 1 - np.exp(-rate_Lambda * x)

# Create DataFrame for better tabular display
df_table = pd.DataFrame({
    'Time (minutes)': x.round(2),
    'PDF': pdf.round(6),
    'CDF': cdf.round(6)
})
print(df_table.to_string(index=False))

# Plotting the PDF and CDF
plt.figure(figsize=(12, 6))
plt.plot(x, pdf, label='PDF', color='blue')
plt.plot(x, cdf, label='CDF', color='orange')
plt.title('Exponential Distribution of Customer Interarrival Times')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
