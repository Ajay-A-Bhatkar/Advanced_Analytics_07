'''
Use the hypothermia dataset. Hypothermia is a medical condition that occurs when the body temperature is below 95 degrees Fahrenheit.
(35 degree Celcius). It is a medical emergency where the body loses heat faster than it can produce it, leading to dangerously low
body temperatures. Patients are treated for this condition. The t.1 Column represents the patient's  body temperature when the patient 
was admitted to the hospital. The t.2 Column represents the patient's body temperature after the initial treatment. Run a paired t-test
to find if the treatment is effective.
'''
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel
from scipy.stats import t

# Load the dataset
df = pd.read_csv('Assignment/Hypothermia.csv')

# We need to group the math scores by parental education level
groups = df.groupby('parental level of education')['math score'].apply(list)

# Extract the groups as separate lists for the ANOVA test
group1 = groups.get('some high school')
group2 = groups.get('high school')
group3 = groups.get('some college')
group4 = groups.get('associate\'s degree')
group5 = groups.get('bachelor\'s degree')
group6 = groups.get('master\'s degree')


# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3, group4, group5, group6)
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
