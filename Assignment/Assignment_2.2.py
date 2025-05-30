'''
Assignment 2: (Conditional Probability)
2) for the combined Marathon datasets... Find the conditional probability of Finish withis 3 hrs given that'Pace < 8 min/mile
Hint:
Split the official time coloumn into hrs, mins, secs
Convert hrs, mins, secs coloumns inito integer
Manually calculate total time in seconds, Repeat for the'Pace coloumns

'''

# Import necessary libraries
import pandas as pd
df_2015 = pd.read_csv('22-05/marathon_results_2015.csv')
df_2016 = pd.read_csv('22-05/marathon_results_2016.csv')
df_2017 = pd.read_csv('22-05/marathon_results_2017.csv')

# Concatenate the datasets into a single DataFrame
df = pd.concat([df_2015, df_2016, df_2017])

# Split the 'Official Time' column into hours, minutes, and seconds
df[['Hours', 'Minutes', 'Seconds']] = df['Official Time'].str.split(':', expand=True)

# Convert hours, minutes, and seconds to integers
df['Hours'] = df['Hours'].astype(int)
df['Minutes'] = df['Minutes'].astype(int)
df['Seconds'] = df['Seconds'].astype(int)

# Calculate total time in seconds
df['Official Time (seconds)'] = df['Hours'] * 3600 + df['Minutes'] * 60 + df['Seconds']

# Split the 'Pace' column into minutes and seconds
df[['Pace Hours','Pace Minutes', 'Pace Seconds']] = df['Pace'].str.split(':', expand=True)

# Convert pace minutes and seconds to integers
df['Pace Hours'] = df['Pace Hours'].astype(int)
df['Pace Minutes'] = df['Pace Minutes'].astype(int)
df['Pace Seconds'] = df['Pace Seconds'].astype(int)

# Calculate total pace in minutes
df['Pace (sec)'] = df['Pace Hours']*3600 + df['Pace Seconds']  + df['Pace Minutes'] * 60



# Calculate the conditional probability of Finish with 3 hrs given that Pace < 8 min/mile
cp_pace_gt_8min = df[(df['Official Time (seconds)'] < 10800) & (df['Pace (sec)'] < 480)]
len_of_cp_pace_gt_8min =len(cp_pace_gt_8min)

# Calculate the total number of rows in the dataset
total = len(df['Pace (sec)']<480)

# Calculate the conditional probability
conditional = len_of_cp_pace_gt_8min/total

# Print the result
print("The conditional probability of Finish with 3 hrs given that Pace < 8 min/mile is:", conditional) 