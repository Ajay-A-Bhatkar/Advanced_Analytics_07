'''
2. For the diabetes dataset, divide people into two groups : 
(a) age <= 40
(b) age > 40
Take 30 samples each run a two sample t-test to see if thier glucose levels have a significant statistical difference.
'''

import pandas as pd
import numpy as np
from scipy import stats, norm

df = pd.read_csv('Assignment\diabetes.csv')
print(df.head())

# a) age <= 40
df1 = df[df['Age'] <= 40]

# b) age > 40
df2 = df[df['Age'] > 40]

np.random.seed(42)
n = 1000
sample_data = np.random.normal(loc=30.1, scale=3, size=n)

mu0 = 30

sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
se = sample_std / np.sqrt(n)

z_stat = (sample_mean - mu0) / se
p_value = norm.cdf(z_stat)

alphas =[0.01, 0.05, 0.10]
print(f"{'Alpha':<8}{'Z-Stat':>10}{'P-Value':>12}{'Critical Z':>15}{'Decision':>15}")
print("-" * 60)

for alpha in alphas:
    critical_z = norm.ppf(1 - alpha) # one tailed critical value
    descision = "Reject H0" if (z_stat) > critical_z else "Fail to Reject H0"
    print(f"{alpha:<8.2f}{z_stat:10.4f}{p_value:12.4f}{critical_z:15.4f}{descision:>15}")