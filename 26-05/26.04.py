
#One - sample test

import pandas as pd
from scipy import stats

data = pd.read_csv('C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/Assignment/Mall_Customers.csv')
age_data = data['Age']

age_data = age_data.sample(n=30, replace=False)

#Hypothesised population average age
pop_avg_age = 40

#Perform one-sample t-test
t_stat, p_val = stats.ttest_1samp(age_data, pop_avg_age)

#Degrees of Freedom
degrees_of_freedom = len(age_data) - 1

#Print the results
print('T-statistic:', t_stat)
print('P-value:', p_val)
print('Degrees of Freedom:', degrees_of_freedom)

#Interpretation
alpha = 0.025
if p_val < alpha:
    print("==================================================")
    print("Reject the Null Hypothesis for mean age = 40")
else:
    print("Fail to Reject the Null Hypothesis for mean age = 40")
