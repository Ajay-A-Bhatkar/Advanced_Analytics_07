#Investment risk analysis using mote carlo

import numpy as np
import matplotlib.pyplot as plt

# Parameters
intial_investment = 10000 # Initial investment amount
mean_return = 0.15  # Mean annual return
std_dev = 0.15 # Standard deviation of returns
num_simulations = 1000  # Number of simulations to run
#Monte carlo Simulation
simulated_end_values = []
for i in range(num_simulations):
    # Generate a random return for the year
    random_return = np.random.normal(mean_return, std_dev)
    print(random_return)
    
    # Calculate the end value of the investment after one year
    end_value = intial_investment * (1 + random_return)
    
    # Store the end value
    simulated_end_values.append(end_value)

print(simulated_end_values)

# Results
mean_end_value = np.mean(simulated_end_values) # Mean of all simulated end values
#Proportion of cases where we have a loss (End value < Initial investment)
risk_of_loss = np.sum(np.array(simulated_end_values) < intial_investment)
print("\n =============================================\n")
print(f"Mean end value after one year: ${mean_end_value:.2f}")
print(f"Probability of loss: {risk_of_loss * 100:.2f}%")

# Plotting the distribution of end values
plt.figure(figsize=(10, 6))
plt.hist(simulated_end_values, bins=30, edgecolor='k', alpha=0.7)
plt.title('Simulated Portfolio End Values')
plt.xlabel('Portfolio Value ($)')
plt.ylabel('Frequency')
plt.show()

#Problem download the data
#pip install yfinance
# import yfinance as yf

# aapl = yf.Ticker("AAPL")
# df = aapl.history(period="max")
# print(df.to_string())

#Use MOnte Carlo to simulate the investment value after 1 year if someone invests $10,000 in stock today




