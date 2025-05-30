#Paired t-test: A company has launched some offer
#H0: Pre and post usage is the same (i.e offer is not effective)

from scipy.stats import ttest_rel
from scipy.stats import t
import pandas as pd

df = pd.read_csv("26-05/cust_seg.csv")
usage_data = df[["pre_usage", "post_usage_2ndmonth"]]

#Perform paired t-test
tstat, pval = ttest_rel(usage_data["pre_usage"], usage_data["post_usage_2ndmonth"])

degrees = len(usage_data["pre_usage"]) - 1

alpha = 0.025
critical_value = t.ppf(1 - alpha, degrees)

print(f"T-statistic: {tstat:.2f}, p-value: {pval:.4f}")
print(f"Critical value (alpha = {alpha}, two-tailed): +- {critical_value:.2f}")

if pval < alpha:
    print("Reject Null Hypothesis")
else:
    print("Fail to reject Null Hypothesis")
