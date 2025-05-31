# Geometric Distribution : How many trials before the first success?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the data
df = pd.read_csv('Advanced_Analytics_07/31-05/tips.csv')

#Define the threshold for a 'generous tipper'
avg_tip  = df['tip'].mean()
generous_tippers = df[df['tip'] > avg_tip]

#Calculate the probability of a generous tipper
p = len(generous_tippers) / len(df)
print(p)

'''
 Probability that a customer giving above average tip = around 49.5%
Expected number of customers until the first genberous tipper (geometric distribution)
As we know: expected value of the geometric distribution is 1/p
'''

expected_trials = 1/p

''' 
Simulate the geomertic distribution for demonstartion
geometric(p) : This function genertaes random values following a geometric distribution, which models the nuymber of trials
(or attempts) required before the first success.
p is the probability of success on each trial
(i.e., the probability of a customer giving a generous tip).
Find "How many customers did we have to wait for the first generous tipper?"
Set the probability of success p as(0.5 to get an expected value of 2 trails)
p=0.99... keep manually changing this value to see the impact of geometric distribution
Simulate the geometric distribution for 1000 trails 

'''
simulated_trails=np.random.geometric(p,size=1000)
print(simulated_trails)

#plot the distribution
plt.hist(simulated_trails,bins=20,density=True)
plt.title("Geometric Distribution: NUmber of customer until a generous tipper")
plt.xlabel("Number of customers")
plt.ylabel("Frequency")
plt.show()

#output the expected trail
print(f"Expected number of customers until to find a generous tipper: {expected_trials:.2f}")
