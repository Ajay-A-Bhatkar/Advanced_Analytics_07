# One - Tailed Test(LEFT)
# Hypothesis Testing example on medicine testing
# Pharma company claim: A new medicine reduces fever duration from 16 weeks to a smaller duration
# H0; Mean fever duration is 16 weeks (meu >= 16)
# H1; Mean fever duration is less than 16 weeks (mue < 16)


import numpy as np
from scipy.stats import norm

# Generate random sample data (e.g. recoveries for 1000 patients)
np.random.seed(42)
n = 1000
sample_data = np.random.normal(loc=15.8, scale=3, size=n)

#H0 mean
mu0 = 16

# Sample stats
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
se = sample_std / np.sqrt(n)

#Z-Test (one-tailed)
z_stat = (sample_mean - mu0) / se
p_value = norm.cdf(z_stat)

#print result table
alphas =[0.01, 0.05, 0.10]
print(f"{'Alpha':<8}{'Z-Stat':>10}{'P-Value':>12}{'Critical Z':>15}{'Decision':>15}")
print("-" * 60)

for alpha in alphas:
    critical_z = norm.ppf(1 - alpha) # Two tailed critical value
descision = "Reject H0" if (z_stat) < critical_z else "Fail to Reject H0"
print(f"{alpha:<8.2f}{z_stat:10.4f}{p_value:12.4f}{critical_z:15.4f}{descision:>15}")