import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats


data=pd.read_csv("24-05/diabetes.csv")

total_bill_series=data['total_bill']

#Normallyt test with Shapiro_Wilk test
shapiro_test = stats.shapiro(total_bill_series)
print("Test statistic (W): ",shapiro_test.statistic)

if shapiro_test.pvalue > 0.05:
    print("Data is normally distributed")
else:
    print("Data is not normally distributed")

# QQ Plot
stats.probplot(total_bill_series, dist='norm', plot=plt)
plt.title('QQ Plot of Total Bill')
plt.xlabel('Theoretical Quantiles (Normal Distribution)')   
plt.ylabel('Ordered Values (Total Bill)')
plt.grid(True)
plt.show()


#-------------------------------------------------------------------------------#
#Central Limit theorem
sample_means = []
n_samples = 100
sample_size = 30

for _ in range (n_samples):
    sample = np.random.choice(total_bill_series, size=sample_size, replace=True)
    sample_means.append(sample.mean())

#Plot the distribution of sample means
plt.hist(sample_means, bins=30, edgecolor='k', alpha=0.6, color='b')
plt.title('Distribution of Sample Means (n=30)')
plt.xlabel('Sample Mean')   
plt.ylabel('Frequency')
plt.show()

#Now Perform Shapiro test on sample means and draw QQ plot of the same
shapiro_test = stats.shapiro(sample_means)