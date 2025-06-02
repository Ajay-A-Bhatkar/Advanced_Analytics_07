# Monte Carlo Simulation
import numpy as np

# Part 1
def roll_dice(num_rolls):
    # Simulation of rolling a dice twice, minimum of 1 and maximum of 6
    # Add the results of the two simulation, so possible values will be:
    # (1,1) or (1,2) .... (6,6) Sum will be between 2 and 12
    # Run multiple times to verify

    return np.random.randint(1, 7, 2)
    # Generate an array of 2 random integers between 1 and 6 (both inclusive) and add them.
print(roll_dice(10))


'''# Part 2
# Someone approaches us saying I will give you 5$ if you get 7 and take 1 $ if you get a number other than 7
How do we know what will happpen? Our own "Monte Carlo Simulation" like function 
'''
def monte_carlo_simulation(runs=1000):
    results = np.zeros(2) # An array, results[0] = 0, results[1] = 0
    for _ in range(runs):
        if roll_dice(1).sum() == 7:
            results[0] += 1
        else:
            results[1] += 1
    return results

#Test 2-3 times and time and calculate how much you will wiuni versus lose
print(monte_carlo_simulation())
print(monte_carlo_simulation())
print(monte_carlo_simulation())

# Part 3 - Now do it 1000 times ... Takes some time
results = np.zeros(1000)
for i in range(1000):
    results[i] = monte_carlo_simulation()[0]
print(results)

# Let us plot the results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(results, bins=20, edgecolor='black', alpha=0.7)
plt.show()

# Our win/loss
print(results.mean())        # Genral mean
print(results.mean()*5)      # What we will get as win on an average
print(results.mean() * 4.75) # Just a marginal change in reward-  see imapact
print(1000 - results.mean() * 4.75) # What we will pay on an average
print(results.mean()/1000) # What we will pay on an average per run

