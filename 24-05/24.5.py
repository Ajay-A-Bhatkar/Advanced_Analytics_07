#PMF and CDF for continuous data
import numpy as np
import pandas as pd
from scipy.stats import norm

df = pd.read_csv('24-05/weight-height.csv')

height_mean = df['Height'].mean()
height_std = df['Height'].std()

#Define the specific range for which to calculate PDF values
start_height = 68
end_height = 70

#Generate small range of height values within the specified interval
num_points = 2
height_values_in_range = np.linspace(start_height, end_height, num_points)

pdf_values_in_range = norm.pdf(height_values_in_range, height_mean, height_std)

for i in range(len(height_values_in_range)):
    print(height_values_in_range[i], pdf_values_in_range[i])

#CDF
prob_range_exact = norm.cdf(end_height, height_mean, height_std)
norm.cdf(start_height, height_mean, height_std)
print(prob_range_exact)


