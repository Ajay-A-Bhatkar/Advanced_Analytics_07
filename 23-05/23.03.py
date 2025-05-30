'''Find the probability that exactly 3 out of 5 students like Python. When it's known that probability of a student liking python is 0.66.
'''
from scipy.stats import binom

#Given values:
n = 5  #Total number of students
k = 3  #Number of students who like python
p = 0.66  #Probability of a student liking python

#Calculate binomial probability
probability = binom.pmf(k, n, p)

print(f"Probability that {k} out of {n} students like python: {probability:.4f}")


#------------------------------------- PRACTICE QUESTION ------------------------------------
'''1. Hospital records show that of patients suffering from a specific disease, 75% die of it. 
What is the probability that of six randomly selected Patients, four will recover? => 0.03'''

#Given values:
n = 6  #Number of patients
k = 4  #Number of patients who recover
p = 0.25  #Probability of a patient recovering

#Calculate binomial probability
probability = binom.pmf(k, n, p)

print(f"Probability that {k} out of {n} patients recover: {probability:.4f}")


'''2. In a random experiment, a coin is taken for tossing and it was tossed exactly 10 times. 
What are the probabilities of obtaining exactly six heads out of total 10 tosses? => 0.2'''

#Given values:
n = 10  #Total number of tosses
k = 6  #Number of heads
p = 0.5  #Probability of getting a head

#Calculate binomial probability
probability = binom.pmf(k, n, p)

print(f"Probability of getting exactly {k} heads out of {n} tosses: {probability:.4f}")


'''3. Normally, 65% of all the students who appear for C-DAC entrance test clear it. 
50 students from a coaching class have appeared for C-DAC March 2024 entrance test. 
a) What is the probability that none of them will clear it? 
b) What is the probability that more than 40 will clear It?  => 1.59e-23, 0.006 (Hint: Use binom.cdf)'''

#Given values:
n = 50  #Total number of students
k = 0  #Number of students who clear the test
p = 0.65  #Probability of a student clearing the test

#Calculate binomial probability
probability = binom.pmf(k, n, p)

print(f"Probability that {k} out of {n} students clear the test: {probability:.4f}")

#Given values:
n = 50  #Total number of students
k = 40  #Number of students who clear the test
p = 0.65  #Probability of a student clearing the test

#CDF(40) will give PMF(1) + PMF(2) + PMF(3) + PMF(4) ... PMF(40)
#1 - CDF(40) will give 2nd answer

probability = 1 - binom.cdf(k, n, p)
print(f"Probability that more than {k} students clear the test: {probability:.4f}")

