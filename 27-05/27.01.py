import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv('27-05/children anemia.csv')
df.dropna(inplace=True)

df = df[['Wealth index combined', 'Anemia']]

contigency_table = pd.crosstab(df['Wealth index combined'], df['Anemia'])
print(contigency_table)

chi2, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contigency_table)

print(chi2)
print(p_value)
print(degrees_of_freedom)
print(expected_counts)

alpha = 0.05
if p_value < alpha:
    print('Reject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')
    