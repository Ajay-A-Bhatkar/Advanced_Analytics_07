# TWO-TAILED TESTS
# Hypothesis Testing example for medicine testing
# Pharma company claim: Fever symptoms are not 16 weeks (Historically they are 16 weeks)
#H0: Mean fever duration is 16 weeks (meu = 16 )
#H1: Mean fever duration is <> 16 weeks (meu <> 16)

import numpy as np
from scipy.stats import norm

# Generate random sample data (e.g. recoveries for 1000 patients)
np.random.seed(42)
n = 1000
# Generate random numbers from a normal distribution
# loc = mean (meu) of the normal distribution (center of the bell curve)
# scale = standard deviation (sigma), controlling the spread or variablity
# n = How many data points / observations

#norm.cdf : what we know = z-score, what we to find out = p-value
#norm.ppf : what we know = p-value, what we to find out = z-score

'''
How to decide ?
>>> 1. If z-statistic (z-score that we have calcullted) < z-critical value (for the given alpha for z-table) 
then we fail to reject the null hypothesis(Do not reject H0)
>>> 2. If z-statistic (z-score that we have calcullted) > z-critical value (for the given alpha for z-table) 
then we reject the null hypothesis (Reject H0)
>>> 3. If p-value (of our z-score) > alpha/2  OR  
       If 2 * p-value (of our z-score) > alpha  then we do not reject the null hypothesis (Do not Reject H0)
       >>> ELSE: Reject the null hypothesis (Reject H0)
'''

sample_data = np.random.normal(loc=15.8, scale=3, size=n)

#H0 mean
mu0 = 16

# Sample stats
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
se = sample_std / np.sqrt(n)

#Z-Test (two-tailed)
z_stat = (sample_mean - mu0) / se

#p-value
p_value = 2 * min(norm.cdf(z_stat), 1 - norm.cdf(z_stat))

#print result table
alphas =[0.01, 0.05, 0.10]
print(f"{'Alpha':<8}{'Z-Stat':>10}{'P-Value':>12}{'Critical Z':>15}{'Decision':>15}")
print("-" * 60)

for alpha in alphas:
    critical_z = norm.ppf(1 - alpha / 2) # Two tailed critical value
    descision = "Reject H0" if abs(z_stat) > critical_z else "   Fail to Reject H0"
    print(f"{alpha:<8.2f}{z_stat:10.4f}{p_value:12.4f}{critical_z:15.4f}{descision:>15}")
