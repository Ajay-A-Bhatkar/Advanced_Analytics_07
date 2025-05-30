'''
For cust_seg.csv, use segment column to identify various custoner segments. perform one-way ANOVA test on latest_non_usage column
for various segments. H0: There is no diference among the customer segments
'''
from scipy.stats import f_oneway
from scipy import stats
import pandas as pd

df=pd.read_csv('26-05/cust_seg.csv')

print(df.segment.value_counts())  # unique segments with customer counts

#separate segments

s1=df.Latest_mon_usage[df.segment ==1]
s2= df.Latest_mon_usage[df.segment ==2]
s3=df.Latest_mon_usage[df.segment ==3]

#ANOVA
fstat,pval = f_oneway(s1,s2,s3)

between_df= len([s1,s2,s3])-1 # Number of groups -1
within_df = len(s1)+len(s2)+len(s3) - len([s1,s2,s3]) #Total Observation - number of groups

alpha=0.025

critical_value = stats.f.ppf(1-alpha,between_df,within_df)

print(f"\n F_statistic: {fstat:.2f},p-value:{pval:.4f}")
print(f"Critical value (alpha = {alpha}):{critical_value:.2f}")

#Interpretation
if pval < alpha:
  print("Reject null hupothesis")
else:
  print("Fail to reject null hypothesis")