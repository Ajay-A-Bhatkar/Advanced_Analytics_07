#Patient average hospital visit duration +/- 5 using Poisson Distribution
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

df = pd.read_csv('24-05/hospital_visits.csv')

df['START'] = pd.to_datetime(df['START'])
df['STOP'] = pd.to_datetime(df['STOP'])

#Calculate the time difference between START and STOP
df['DURATION'] = (df['STOP'] - df['START']).dt.total_seconds() / 3600  # Convert to hours
print(df)

#Calculate average events per time unit (here, hours)
avg_duration = df['DURATION'].mean()
print(f"Average duration: {avg_duration} hours") 

#Define range for Poisson distribution (e.g., around the average)
# Will have start and end of the range as two values
k = range(int(avg_duration) - 5, int(avg_duration) + 6)  #+/- 5 hrs from average
probabilities = poisson.pmf(k, avg_duration)

#Print probabilities
print("Poisson probabilities for each k:")
for duration, prob in zip(k, probabilities):
    print(f"Duration: {duration} hours, {prob}")

#Plot Poisson distribution
plt.figure(figsize=(10, 6))
plt.plot(k, probabilities, label = 'Poisson Distribution', marker='o', color='blue')
plt.xlabel('Duration (hours)')  
plt.ylabel('Probability')
plt.title('Poisson Distribution of Hospital Visit Duration')
plt.legend()
plt.grid(True)
plt.show()

      