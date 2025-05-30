#Assignment 2: (Conditional Probability)
# 1) for diabetes.csv dataset, find the probability that a person glucose levels are > 110 given that the person BMI >=25


#Impoting the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the diabetes.csv file
df = pd.read_csv('C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/Assignment/diabetes.csv')

#Calculating the probability
prob = (df.loc[(df['BMI'] >= 25) & (df['Glucose'] > 110), 'BMI'].count()) / (df.loc[(df['BMI'] >= 25), 'BMI'].count())

#Printing the probability
print(prob)