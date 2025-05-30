#EDA of MSD's ODI Career

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('MS_Dhoni_ODI_record.csv')

#Basic Checks
print(df.head())
print(df.tail())

#Data Cleaning - Opposition name says 'v Aus' etc, we can remove 'v '
# df['Opposition'] = df['Opposition'].apply(lambda x: x[2:])
df['opposition'] = df['opposition'].str.replace('v ', '', regex=False)
# regex = False means that the first ('v ') is not a regex, but a literal string

#Add a 'feature' - 'year' column using the match date column
# Convert date column into datetime format
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['year'] = df['date'].dt.year.astype(int)
#print(df.head())

#Create a column to differentiate between out and not out (To calc. avg)
#Count the no . of not outs (ending with '*')
#The apply method in Pandas is used to apply a function to each element of a Series or DataFrame. In this
#case, the apply method is used to create a new column called 'not_out' in the DataFrame df.

df['score'] = df['score'].apply(str)
df['not_out'] = np.where(df['score'].str.endswith('*'), 1, 0)
# lambda x: 1 if x.endswith('*') else 0: This is a lambda function that takes a value x

#Dropping the odi_number, as it adds no value to analysis
df.drop(columns='odi_number', inplace=True)

#Dropping those inningses where the batsman did not bat and storing into new df
# Take all the columns, starting with runs_scored
df_new = df.loc[((df['score'] != 'DNB') & (df['score'] != 'TDNB')), 'runs_scored':]
#print(df_new.head())

#fixing the data types of numerical columns
df_new['runs_scored'] = df_new['runs_scored'].astype(int)
df_new['balls_faced'] = df_new['balls_faced'].astype(int)
df_new['strike_rate'] = df_new['strike_rate'].astype(float)
df_new['fours'] = df_new['fours'].astype(int)
df_new['sixes'] = df_new['sixes'].astype(int)

# Career stats
first_match_date = df['date'].dt.date.min().strftime('%B %d, %Y') # first match
print('First match:', first_match_date)
last_match_date = df['date'].dt.date.max().strftime('%B %d, %Y') # last match
print('Last match:', last_match_date)
number_of_matches = df.shape[0] # number of mathces played in career
print('Number of matches played:', number_of_matches)
number_of_inns = df_new.shape[0] # number of innings
print('Number of innings played:', number_of_inns)
not_outs = df_new['not_out'].sum() # number of not outs in career
print('Not outs:', not_outs)
runs_scored = df_new['runs_scored'].sum() # runs scored in career
print('Runs scored in career:', runs_scored)
balls_faced = df_new['balls_faced'].sum() # balls faced in career
print('Balls faced in career:', balls_faced)
career_sr = (runs_scored / balls_faced)*100 # career strike rate
print('Career strike rate: {:.2f}'.format(career_sr))
career_avg = (runs_scored / (number_of_inns - not_outs)) # career average
print('Career average: {:.2f}'.format(career_avg))
highest_score = df_new['runs_scored'].max()
print('Highest score in career:', highest_score)
hundreds = (df_new['runs_scored'] >= 100).sum()
print('Number of 100s:', hundreds)
fifties = ((df_new['runs_scored'] >= 50) & (df_new['runs_scored'] < 100)).sum()
print('Number of 50s:', fifties)
fours = df_new['fours'].sum() # number of fours in career
print('Number of 4s:', fours)
sixes = df_new['sixes'].sum() # number of sixes in career
print('Number of 6s:', sixes)

#number of matches played against different oppositions, count the occurences of each unique value in the 
#'opposition' column. Opposition_counts will be a series with labelled index as opposition.
opposition_counts = df['opposition'].value_counts()
print(opposition_counts)

#Plot the counts as bar plot
opposition_counts.plot(kind='bar', title='Number of matches against different oppositions', figsize=(10, 6))
plt.show()

#Runs scored against each team
# Group the DataFrame by 'opposition' column
grouped_by_opposition = df_new.groupby('opposition')
# Sum the 'runs_scored' column for each group
sum_of_runs_scored = grouped_by_opposition['runs_scored'].sum()
print(sum_of_runs_scored)
#Sum_of_run_scored is a series with labelled index as opposition.
#convert it into df and remove the index

runs_scored_by_opposition = pd.DataFrame(sum_of_runs_scored).reset_index()
runs_scored_by_opposition.plot(x='opposition', kind='bar', title='Runs scored against each team', figsize=(10, 6))
plt.xlabel(None)
plt.show()

#Does not look good... Let us sort it ....

sorted = runs_scored_by_opposition.sort_values(by='runs_scored', ascending=False)
#Plot the sorted data
sorted.plot(x='opposition', kind='bar', title='Runs scored against each team', figsize=(10, 6))
plt.xlabel(None)
plt.show()

#Boxplot of runs against various oppositions

sns.boxplot(x='opposition', y='runs_scored', data=df_new)
plt.show()

#Looks crowded - Let's retain only major countries
# List of oppostion to filter

opposition_list = ['Australia', 'England', 'South Africa', 'Pakistan', 'Sri Lanka', 
                   'West Indies', 'New Zealand', 'Bangladesh']

# Filter rows where 'opposition' is in the list
df_filtered = df_new[df_new['opposition'].isin(opposition_list)]

#Sort the filtered dataframe by 'runs_scored' in descending order of 'runs_scored'
df_filtered = df_filtered.sort_values(by='runs_scored', ascending=False)

#Redraw the boxplot but on filtered opposition list
sns.boxplot(x='opposition', y='runs_scored', data=df_filtered)
plt.xticks(rotation=45)
plt.show()

# histogram (distplot) with  and without kde (kernel density estimation)
sns.displot(data=df_filtered, x='runs_scored', kde=False)
plt.show()

#We see that there is right/positive skewness

sns.displot(data=df_filtered, x='runs_scored', kde=True)
plt.show()

# Histogram with bins
sns.set(style = 'darkgrid')
sns.histplot(data=df_new, x='runs_scored', bins=15)
plt.show()

#joinplot
sns.jointplot(x='ball_faced', y='runs_scored', data=df_new, kind='scatter')
plt.show()

#Heatmap
#Calculation the correlation matrix
correlation_matrix = df_new[['runs_scored', 'balls_faced', 'fours', 'sixes']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(data=correlation_matrix, annot=True, cmap='viridis',
            square=True, fmt= ".2f")
plt.title('Correlation Heatmap between Balls Faced, Runs Scored')
plt.show()

#Yearwise record
year_counts = df_new['year'].value_counts()
sorted_counts = year_counts.sort_index()
sorted_counts.plot(kind='bar', title='Yearwise record', figsize=(10, 6))
plt.xticks(rotation=0)
plt.show()

#Runs scored by year
grouped_by_year = df_new.groupby('year')
sum_of_runs_scored = grouped_by_year['runs_scored'].sum()
df_grouped = sum_of_runs_scored.reset_index()
plt.figure(figsize=(10, 6))
x_values = df_grouped['year']
y_values = df_grouped['runs_scored']
plt.bar(x_values, y_values, color='skyblue', edgecolor='black')
plt.title('Bar Plot of Runs scored by year')
plt.show()
