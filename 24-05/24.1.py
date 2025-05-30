# Poisson Distribution: Historically, 8 defective products are manufactured per day. 
# What is the probability of exactly 4 defective products are manufactured in a day?

import math
# define the parameters
lamda = 8
x = 4
p_4 = (math.exp(-lamda) * (lamda ** x) / math.factorial(x))
print(p_4)

#Using Poisson Distribution formula
from scipy.stats import poisson
mu = 8
x = 4
print(poisson.pmf(x, mu))