import numpy as np
import pandas as pd
import math as m

#Marginal and Joint(Unconditional) Probability

df = pd.read_csv("22-05/Titanic-Dataset.csv")
#Select relevant columns
df = df[['Sex', 'Survived']]
df.dropna(inplace=True)

#Total number of rows
total = len(df)
print(f"Total number of passengers: {total}")

#Total number of males and females
p_male = len(df[df['Sex'] == 'male']) / total
p_female = len(df[df['Sex'] == 'female']) / total

#Total number of pasengers who Survived
p_survived = len(df[(df['Survived'] == 1)]) / total
p_not_survived = len(df[(df['Survived'] == 0)]) / total

print("\nMarginal Probability : ")
print(f"P(Sex = male) : {p_male:.3f}")
print(f"P(Sex = female) : {p_female:.3f}")
print(f"P(Survived = 1) : {p_survived:.3f}")
print(f"P(Survived = 0) : {p_not_survived:.3f}")


#Joint Probabilities

#Male and Survived
p_male_survived = len(df[(df['Sex'] == 'male') & (df['Survived'] == 1)]) / total
p_male_not_survived = len(df[(df['Sex'] == 'male') & (df['Survived'] == 0)]) / total

#Female and Survived
p_female_survived = len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)]) / total
p_female_not_survived = len(df[(df['Sex'] == 'female') & (df['Survived'] == 0)]) / total

print("\nJoint Probability : ")
print(f"P(Sex = male, Survived = 1) : {p_male_survived:.3f}")
print(f"P(Sex = male, Survived = 0) : {p_male_not_survived:.3f}")
print(f"P(Sex = female, Survived = 1) : {p_female_survived:.3f}")
print(f"P(Sex = female, Survived = 0) : {p_female_not_survived:.3f}")

#Conditional Probability
# P(Survive | Female) = P(Survived and Female) / P(female)\
p_survived_given_female = p_female_survived / p_female
print(f"P(Survived | Female) : {p_survived_given_female:.3f}")

p_survived_given_male = p_male_survived / p_male
print(f"P(Survived | Male) : {p_survived_given_male:.3f}")

# p_female_survived: This is a joint probability. It answers:
# "What is the probability that a randomly selected passenger was female and survived?"
# Calculated as : len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)]) / total

#p_survived_given_female: This is a conditional probability. It answers:
# "What is the probability that a randomly selected passenger who survived was female?"
# Calculated as : p_female_survived / p_female
print("\nConditional Probability : ")
print(f"P(Survived | Female) : {p_survived_given_female:.3f}")
print(f"P(Survived | Male) : {p_survived_given_male:.3f}")



