'''
5) Use Dataset airline_passenger_satisfaction.csv. For male passengers, determine if passenger class and ratings
   given to "onboard services" are independent. Separately repeat for female passengers.
'''
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('Assignment/airline_passenger_satisfaction.csv')
df.dropna(inplace=True) 

df_male = df[df['Gender'] == 'Male']
df_male_new = df_male[['Class','Satisfaction']]

contigency_table = pd.crosstab(df_male_new['Class'], df_male_new['Satisfaction'])
print(contigency_table)

chi2, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contigency_table)

print(f"Chi-square: ",chi2)
print(f"P-Value: ",p_value)
print(f"Degrees of Freedom: ",degrees_of_freedom)
print(expected_counts)

alpha = 0.05
if p_value < alpha:
    print('Reject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')


df_female = df[df['Gender'] == 'Female']
df_female_new = df_female[['Class','Satisfaction']]

contigency_table = pd.crosstab(df_female_new['Class'], df_female_new['Satisfaction'])
print(contigency_table)

chi2, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contigency_table)

print(f"Chi-square: ",chi2)
print(f"P-Value: ",p_value)
print(f"Degrees of Freedom: ",degrees_of_freedom)
print(expected_counts)

alpha = 0.05
if p_value < alpha:
    print('Reject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')
