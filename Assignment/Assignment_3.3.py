'''
3. Use billionaires dataset. Calculate Z-Scores (FinalWorth Column), Find and plot the outliers. 
'''

import pandas as pd
from scipy.stats import zscore

df = pd.read_csv('Assignment/Billionaires Statistics Dataset.csv')

FinalWorth_mean = df['finalWorth'].mean()
FinalWorth_std = df['finalWorth'].std()


# Calculate the z-scores for FinalWorth
df['z-score'] = zscore(df['finalWorth'])

#Display mean and SD of FinalWorth column
print("Mean of FinalWorth column: ", FinalWorth_mean)
print("Standard Deviation of FinalWorth column: ", FinalWorth_std)

#Display the first few rows of the dataframe with the z-scores
print("\n First few rows of the dataframe with the z-scores:")
print(df.head())
print(len(df))

#outlier analysis
zscore_threshold = 3

# Find the rows where the Z-Score is greater than the threshold
FinalWorth_outliers = df[(df['z-score'].abs() > zscore_threshold)]
# weight_outliers = df[(df['Weight_zscore'].abs() > zscore_threshold)]
print(f"\nNumber of outliers: (z-score > {zscore_threshold}): {len(FinalWorth_outliers)}")

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

#Plot for FinalWorth
plt.subplot(1, 1, 1) # 1 row, 1 columns, 1st plot
sns.histplot(df['finalWorth'], kde=True, color='skyblue')
plt.scatter(FinalWorth_outliers['finalWorth'], [0] * len(FinalWorth_outliers), color='red', s=50, zorder=5, label=f'{zscore_threshold} z-score outliers')
plt.axvline(x=FinalWorth_mean, color='green', linestyle='--', label=f'Mean: {FinalWorth_mean:.2f}')
plt.title('FinalWorth Distribution by outliers')
plt.xlabel('FinalWorth')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True,linestyle='--', alpha=0.7)
plt.show()





