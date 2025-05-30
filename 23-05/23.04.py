'''
Imagine we are observing how people tip at a restaurant. We want to understand how often people give a generous tip - say, more 
than 15% of their total bill.

We have got real data from many customers, and based on that data, we estimate the probability that a 
randomly chosen customer tips more than 1.5%.
Now, suppose we randomly pick 100 customers then are likely to be generous tippers? we want to know how many of

Will it be 40? 60? 80?

Binomial distribution helps us figure this out
Shows us the full range of possibilities from 0 to 100 generous
tippers and tells us how likely each number is, based on the actual data Analogy: 
Every customer tosses a coin and the result is "Generous tipper or "not" - 
Based on the data we know how likely that the coin lants on "Generous Now what happens if we toss the coin 100 times? How many times will we get "Generous"?

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

tips_df = pd.read_csv('23-05/tips.csv')

'''df1=df[df['tip']/df['total_bill'] > 0.15]
p=len(df1)/len(df)
'''
#Binary outcome: Success if (tip/total_bill) > 0.15
def classify_tip(row):
    total_bill = row['total_bill']
    tip = row['tip']
    if (tip / total_bill) > 0.15:
        return 1
    else:
        return 0
#Apply the function to create the binary column
tips_df['tip_binary'] = tips_df.apply(classify_tip, axis=1)

#Total number of trials (total observations)
n = len(tips_df)

#Number of successes (tips where tip / total_bill > 0.15)
k = tips_df['tip_binary'].sum()

# probability of success (p)
p = k/n
 #Define number of trials for binomial distribution
trials = 100  

#Binomial probability of exactly k successes in trials
hypothetical_k = 50
probability = binom.pmf(hypothetical_k, trials, p)

#Exact binomial probability of 60% success (60 out of 100 trials)
min_desired_successes = int (0.6 * trials)
exact_probability_60 = binom.pmf(min_desired_successes, trials, p)

#Just to calculate it at 65% success
exact_probability_65 = binom.pmf(65, trials, p)

#Cummulative probability of 60% success (60 out of 100 trials)
cumulative_prob = binom.cdf(min_desired_successes, trials, p)

#Output the results
print(f"Total Trials (n): {n}")
print(f"Number of Successes (k): {k}")
print(f"Probability of Success (p): {p:.4f}")
print(f"Binomial Probability of exactly {hypothetical_k} successes in {trials} trials:.4f")
print(f"Exact Binomial Probability of 60% success (60 out of {trials} trials): {exact_probability_60:.4f}")
print(f"Cummulative probability for 60 or fewer successes in {trials} trials: {cumulative_prob:.4f}")
print(f"Exact Binomial Probability of 65% success (65 out of {trials} trials): {exact_probability_65:.4f}")

#Generate X values (number of successes from 0 to 100)
x = np.arange(0, trials +1)

#Calculate the PMF for each number of successes
pmf_values = binom.pmf(x, trials, p)

#Calculate the CDF for each number of success
cdf_values = binom.cdf(x, trials, p)

#Plotting the PMF and CDF
plt.figure(figsize=(10, 6))

#PMF Plot
plt.subplot(1,2,1)
plt.bar(x, pmf_values, color='blue', alpha=0.7)
plt.title('Binomial Distribution PMF (n={trials}, p={p:.4f})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.xticks(np.arange(0, trials +1, 5))
plt.grid(axis='y', linestyle = '--', alpha=0.7)
plt.axvline(x=min_desired_successes, color='red', linestyle='--', label='60 Success')
plt.legend()

#CDF Plot
plt.subplot(1,2,2)
plt.plot(x, cdf_values, color='orange', label='CDF')
plt.title('Binomial Distribution CDF (n={trials}, p={p:.4f})')
plt.xlabel('Number of Successes')
plt.ylabel('Cummulative Probability')
plt.xticks(np.arange(0, trials +1, 5))
plt.grid()
plt.axhline(y = cumulative_prob, color='blue', linestyle='--', label='60 Success')
plt.legend()
plt.tight_layout()
plt.show()
