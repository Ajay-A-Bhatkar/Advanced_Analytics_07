'''
Practice questions (No need to solve using formula, use functions only):

1. Normally, a salesperson has to drive through 3 traffic signals during a sales trip. 
What is the probability that he will cross (a) 4 traffic signals and (b) up to 4 traffic signals today?

2. A life insurance salesman sells on the average 3 life insurance policies per week. 
Use Poisson's law to calculate the probability

a. In a given week he will sell some policies

b. In a given week, he will sell 2 or more policies but not more than five policies.

c. Assuming that per week, there are 5 working days, what is the probability that on a given day, he will sell 1 policy?

'''

from scipy.stats import poisson
x = 4
lam = 3
prob_a = poisson.pmf(x, lam)
print(prob_a)

prob_b = poisson.cdf(4, lam)
print("**************************************")
# print(prob_b)

# 2.a
lam = 3
prob_a = 1 - poisson.cdf(0, lam)
print("********** 2.a **********")
print(prob_a)

# 2.b
lam = 3
prob_b = poisson.cdf(5, lam) - poisson.cdf(2, lam)
print(sum(poisson.pmf(i, lam) for i in range(2, 6)))
print("********** 2.b **********")
print(prob_b)
print(prob_b)

#2.c
x = 1
lam = 3/5
prob_c = poisson.pmf(x, lam)
print("********** 2.c **********")
print(prob_c)