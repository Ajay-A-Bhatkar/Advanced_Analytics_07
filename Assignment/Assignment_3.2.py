'''
2. Use titanic dataset. a)Calculate the historical probability of any passenger surviving. 
   b)Use binomial distribution to calculate and plot probability of 
   exact 0 out of random 100 passengers surviving, exactly 1 out of random 100 passengers surviving,
   exactly 2 out of random 100 passengers surviving, ... up to all 100 passengers surviving.
'''
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("22-05/Titanic-Dataset.csv")

Total_Passengers = len(df)  #To get total passengers on Titanic
print("Total passengers on Titanic: ", Total_Passengers)

df_survived = df[df["Survived"] == 1]  #To get total number of survived passengers
Survived_Passengers = len(df_survived)
print("No. of passengers who survived: ", Survived_Passengers)

#a)Calculate the historical probability of any passenger surviving. 
Survival_Probability = Survived_Passengers / Total_Passengers
print("Historical probability of any passenger surviving: ", Survival_Probability)


