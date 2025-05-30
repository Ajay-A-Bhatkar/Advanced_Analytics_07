#Z-Score 
import pandas as pd
from scipy.stats import zscore
df = pd.read_csv('24-05/weight-height.csv')

weight_mean = df['Weight'].mean()
weight_std = df['Weight'].std()
height_mean = df['Height'].mean()
height_std = df['Height'].std()

# Calculate the z-scores for the weight and height
df['Weight_zscore'] = zscore(df['Weight'])
df['Height_zscore'] = zscore(df['Height'])

#Display the mean and SD for weight and height
print(f"Mean weight: {weight_mean:.2f}")
print(f"Standard deviation of weight: {weight_std:.2f}")
print(f"Mean height: {height_mean:.2f}")
print(f"Standard deviation of height: {height_std:.2f}")

#Display the first few rows of the dataframe with the z-scores
print("\n First few rows of the dataframe with the z-scores:")
print(df.head())

#Save the modified dataframe to a CSV file
# df.to_csv('24-05/weight-height_zscore.csv')

#outlier analysis
zscore_threshold = 3

weight_outliers = df[(df['Weight_zscore'].abs() > zscore_threshold)]
height_outliers = df[(df['Height_zscore'].abs() > zscore_threshold)]
all_outliers = pd.concat([weight_outliers, height_outliers]).drop_duplicates()

print(f"\nNumber of weight outliers: (z-score > {zscore_threshold}): {len(weight_outliers)}")
print(f"Number of height outliers: (z-score > {zscore_threshold}): {len(height_outliers)}")
print(f"Number of all outliers: (z-score > {zscore_threshold}): {len(all_outliers)}")

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

#Plot for weight
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
sns.histplot(df['Weight'], kde=True, color='skyblue')
plt.scatter(weight_outliers['Weight'], [0] * len(weight_outliers), color='red', s=50, zorder=5, label=f'{zscore_threshold} z-score outliers')
plt.axvline(x=weight_mean, color='green', linestyle='--', label=f'Mean: {weight_mean:.2f}')
plt.title('Weight Distribution by outliers')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True,linestyle='--', alpha=0.7)

#plot for height
plt.subplot(1, 2, 2) # 1 row, 2 columns,
sns.histplot(df['Height'], kde=True, color='lightgreen')
plt.scatter(height_outliers['Height'], [0] * len(height_outliers), color='red', s=50, zorder=5, label=f'{zscore_threshold} z-score outliers')
plt.axvline(x=height_mean, color='blue', linestyle='--', label=f'Mean: {height_mean:.2f}')
plt.title('Height Distribution by outliers')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.legend()  
plt.grid(True,linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------#

