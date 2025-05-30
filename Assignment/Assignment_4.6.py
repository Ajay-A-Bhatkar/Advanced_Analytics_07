'''
6) Use dataset Amazon Sale Report.
   Are order fulfillment and order status independent ? Comsider only cancelled and pending orders.
   ''' 
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('Assignment/Amazon Sale Report.csv')
'''df.dropna(inplace=True) ''' #not using drop.na function as it will also remove cancelled orders.

#Filter the dataset to include only cancelled and pending orders
df_new = df[(df['Status'] == 'Cancelled') | (df['Status'] == 'Pending')]

# Print the filtered dataset
print(df_new.head)

# Create a contingency table
contingency_table = pd.crosstab(df_new['Fulfilment'], df_new['Status'])
print(contingency_table)

chi2, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contingency_table)

print(f"Chi-square: ",chi2)
print(f"P-Value: ",p_value)
print(f"Degrees of Freedom: ",degrees_of_freedom)
print(expected_counts)

alpha = 0.05
if p_value < alpha:
    print('Reject the null hypothesis')
else:
    print('Fail to reject the null hypothesis')