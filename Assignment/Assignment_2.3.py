#Assignment 2: Bayes Theorem (Nobel.csv)
#1. Find the conditional probability of winning a nobel prize in physics, given the laureate is female.
#2. Find the conditional probability of winning a Nobel prize in Literature, given the laureate was born in USA.
#------------------------------------------------------------------------------------------------------------------------------
import pandas as pd

#Read the data from the CSV file
df = pd.read_csv('Assignment/nobel.csv')

#Total Nobel Prize winners
N = len(df)

#Winners in Physics
df_physics = df[df['category'] == 'Physics']
N_physics = len(df_physics)

#Female Laureates
df_female = df[df['gender'] == 'female']
N_female = len(df_female)

#Laureates in Literature
df_literature = df[df['category'] == 'Literature']
N_literature = len(df_literature)

#Laureates born in USA
df_usa = df[df['birth_country'] == 'USA']
N_usa = len(df_usa)

#female laureates in physics
df_female_physics = df[(df['category'] == 'Physics') & (df['gender'] == 'female')]
N_female_physics = len(df_female_physics)

#US Born winners in literature
df_usa_literature = df[(df['category'] == 'Literature') & (df['birth_country'] == 'USA')]
N_usa_literature = len(df_usa_literature)

#Calculation of Individual Probabilities
P_win_physics = (N_physics / N)
P_female = (N_female / N)
P_female_physics = (P_female * P_win_physics)/ P_win_physics

P_usa = (N_usa / N)   #Probability of being a laureate born in USA
P_win_literature = (N_literature / N)  #Probability of winning a Nobel Prize in Literature
P_usa_literature = (P_usa * P_win_literature)/ P_win_literature  #Probability of being a laureate born in USA and winning a Nobel Prize in Literature


'''print("Probability of winning a Nobel Prize in Physics: ", P_win_physics)
print("Probability of being a female laureate: ", P_female)
print("Probability of being a female laureate in Physics: ", P_female_physics)'''

#Bayes' Theorem to calculate the probability of winning a Nobel Prize in Physics given that the laureate is female

N_physics_female = len(df_physics[df_physics['gender'] == 'female'])
P_win_physics_female =(P_female_physics * P_win_physics)/ P_female

print("Probability of winning a Nobel Prize in Physics given that the laureate is female: ", P_win_physics_female)

#Bayes ' Theorem to calculate the probability of winning a Nobel Prize in Literature given that the laureate was born in USA
'''print("Probability of being a laureate born in USA: ", P_usa)
print("Probability of winning a Nobel Prize in Literature: ", P_win_literature)'''

N_literature_usa = len(df_usa_literature)
P_win_literature_usa = (P_usa_literature * P_win_literature)/ P_usa


print("Probability of winning a Nobel Prize in Literature given that the laureate was born in USA: ", P_win_literature_usa)

