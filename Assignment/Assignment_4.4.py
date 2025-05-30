'''
4) For StudentsPerformance.csv using ANOVA find if there is a significant difference in Maths marks  
   depending on the level of parental education   
'''

from scipy.stats import f_oneway
from scipy import stats
import pandas as pd

df=pd.read_csv('Assignment/StudentsPerformance.csv')

# #separate segments

s1=df['math score'][df['parental level of education'] =="bachelor's degree"]
s2=df['math score'][df['parental level of education'] =="some college"]
s3=df['math score'][df['parental level of education'] =="master's degree"]
s4=df['math score'][df['parental level of education'] =="associate's degree"]
s5=df['math score'][df['parental level of education'] =="high school"]
s6=df['math score'][df['parental level of education'] =="some high school"]


# #ANOVA
fstat,pval = f_oneway(s1,s2,s3,s4,s5,s6)

between_df= len([s1,s2,s3,s4,s5,s6]) - 1 # Number of groups -1
within_df = len(s1)+len(s2)+len(s3)+len(s4)+len(s5)+len(s6) - len([s1,s2,s3,s4,s5,s6]) #Total Observation - number of groups

alpha=0.025

critical_value = stats.f.ppf(1-alpha,between_df,within_df)

print(f"\n F_statistic: {fstat:.2f} , p-value:{pval:4f}")
print(f"Critical value (alpha = {alpha}):{critical_value:.2f}")

# #Interpretation
if pval < alpha:
  print("Reject null hupothesis")
else:
  print("Fail to reject null hypothesis")