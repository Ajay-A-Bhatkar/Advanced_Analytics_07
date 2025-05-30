import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats


data=pd.read_csv("C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/24-05/tips.csv")

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

#Rarely ... Data is normally distributed
#Change dataset name and column name to below and retry
# data = pd.read_csv("jersey-rainfall-1984-to-2021.csv")
# total_bill_series = data[YearTotalmm]
