#Naive Bayes Theorem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('29-05/Titanic-Dataset.csv')

#Drop rows with missing values
df = df.dropna(subset=['Sex', 'Pclass', 'Survived'])

total = len(df)

#prior probability of survival
p_Survived_1 = len(df[df['Survived']==1]) / total
p_Survived_0 = len(df[df['Survived']==0]) / total

#  Conditional Probability of survival given Sex
# P(Sex
#='female' | Survived=1)
p_female_given_1 = len(df[(df['Sex']=='female') &  (df['Survived']==1)]) / len(df[df['Survived']==1])

# P(Pclass=2 | Survived=1)
p_Pclass_2_given_1 = len(df[(df['Pclass']==2) &  (df['Survived']==1)]) / len(df[df['Survived']==1])

# P(Sex
#='female' | Survived=0)
p_female_given_0 = len(df[(df['Sex']=='female') &  (df['Survived']==0)]) / len(df[df['Survived']==0])

# P(Pclass=2 | Survived=0)
p_Pclass_2_given_0 = len(df[(df['Pclass']==2) &  (df['Survived']==0)]) / len(df[df['Survived']==0])

# Naive Bayes Theorem : P(x|Survived=1) * P(Survived=1) / P(x)
px_given_1 = p_female_given_1 * p_Pclass_2_given_1 * p_Survived_1

# Naive Bayes Theorem : P(x|Survived=0) * P(Survived=0) / P(x)
px_given_0 = p_female_given_0 * p_Pclass_2_given_0 * p_Survived_0

#Normalize to get probabilities
# Example : px_given_0 = 0.2  and px_given_1 = 0.1 ... then total_prob = 0.3
# So, final_Survived_0 = 0.2 / 0.3 = 0.6666 and final_Survived_1 = 0.1 / 0.3 = 0.3333
# So. now both classes add up to 1 or 100%
total_prob = px_given_0 + px_given_1
final_Survived_0 = px_given_0 / total_prob
final_Survived_1 = px_given_1 / total_prob

print(f"P(Survived=0 | x) = {final_Survived_0:.4f}")
print(f"P(Survived=1 | x) = {final_Survived_1:.4f}")