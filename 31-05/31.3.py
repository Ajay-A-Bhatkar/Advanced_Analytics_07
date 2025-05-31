'''
2.predict how long(i.e. how many records/rows) will we need to wait till we see a record 
of a passenger who survived in the titanic dataset.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# 1. Load the Titanic dataset
df = pd.read_csv('Advanced_Analytics_07/31-05/Titanic-Dataset.csv')

# 2. Calculate the probability of survival
p = df['Survived'].mean()

# 3. Expected number of records until first survivor
expected_records = 1 / p

# 4. Simulate the waiting time
simulated_wait = np.random.geometric(p, size=1000)

# 5. Print results
print(f"Probability of survival: {p:.4f}")
print(f"Expected number of records until first survivor: {expected_records:.2f}")
print(f"Average simulated wait: {simulated_wait.mean():.2f} records")

# 6. Plot the geometric distribution
x = np.arange(1, 30)
y = geom.pmf(x, p)
plt.plot(x, y)
plt.xlabel('Number of records')
plt.ylabel('Probability')
plt.title('Geometric Distribution: Waiting for a Survivor')
plt.grid(True)
plt.show()
