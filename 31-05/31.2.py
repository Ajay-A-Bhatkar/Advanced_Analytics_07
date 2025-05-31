'''
1.for yesterdays combined cricket dataset,predict the 1st instance when your favourite batsman will score a half century at least.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

df_1 = pd.read_csv('Advanced_Analytics_07/Assignment/Men ODI Player Innings Stats - 20th Century.csv')
df_2 = pd.read_csv('Advanced_Analytics_07/Assignment/Men ODI Player Innings Stats - 21st Century.csv')

df_combined = pd.concat([df_1, df_2])

# Convert 'Innings Runs Scored' to numeric
df_combined['Innings Runs Scored'] = pd.to_numeric(df_combined['Innings Runs Scored'], errors='coerce')

fav_player = "MS Dhoni"  # Check exact spelling in your data

# Filter for favorite batsman
df_fav = df_combined[df_combined['Innings Player'] == fav_player]

if len(df_fav) == 0:
    print("Player not found. Check spelling or data.")
else:
    # Filter for half-centuries
    half_centuries = df_fav[df_fav['Innings Runs Scored'] >= 50]

    if len(df_fav) > 0:
        p = len(half_centuries) / len(df_fav)
        expected_innings = 1 / p

        # Simulate the number of innings until first half-century
        simulated_innings = np.random.geometric(p, size=1000)

        # Plot the distribution
        plt.hist(simulated_innings, bins=20, density=True)
        plt.title("Geometric Distribution: Number of Innings Until First Half-Century")
        plt.xlabel("Number of innings")
        plt.ylabel("Frequency")
        plt.show()

        # Plot the geometric PMF
        x = np.arange(1, 100)
        y = geom.pmf(x, p)
        plt.plot(x, y)
        plt.xlabel('Number of innings')
        plt.ylabel('Probability')
        plt.title('Geometric Distribution')
        plt.show()

        print(f"Expected number of innings until first half-century: {expected_innings:.2f}")
    else:
        print("No valid innings data for the player.")
