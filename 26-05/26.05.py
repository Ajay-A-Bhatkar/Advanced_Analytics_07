'''Consider dataset Mall_customer.csv. 
The Genre coloumn can have either Male or Female as value . 
Run a two-sample t-test on the 'Spending score (1 - 100)' column to see if the two genres spend differently
'''
#Two independent samples t-test
#H0: There is no difference in spending score between Male and Female customer spending 

import pandas as pd
from scipy import stats

data = pd.read_csv('Assignment\Mall_Customers.csv')

male_spending = data[data['Genre'] == 'Male']['Spending Score (1-100)']
female_spending= data[data['Genre'] == 'Female']['Spending Score (1-100)']

#perform two-sample t-test
t_statistics, p_value = stats.ttest_ind(male_spending, female_spending)

n_male = len(male_spending)
n_female = len(female_spending)
degrees_of_freedom = n_male + n_female - 2

alpha = 0.05

critical_value = stats.t.ppf(1 - alpha, degrees_of_freedom)

print("T-statistics:", t_statistics)
print("P-value:", p_value)
print("Critical value:", critical_value)


if p_value < alpha:
    print("Reject H0: There is a difference in spending score between Male and Female customer spending")
else:           
    print("Fail to reject H0: There is no difference in spending score between Male and Female customer spending")
