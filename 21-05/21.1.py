import matplotlib.pyplot as plt
import seaborn as sns

# Data
commuter_times = [16, 8, 35, 17, 13, 15, 15, 5, 16, 25, 20, 20, 12, 10]

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=commuter_times, orient='h')

# Add titles and labels
plt.title('Box Plot of Commuter Times')
plt.xlabel('Minutes')

# Show the plot
plt.show()
