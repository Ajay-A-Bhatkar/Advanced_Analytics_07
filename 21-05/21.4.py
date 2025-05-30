# correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/dbda.STUDENTSDC/Desktop/Advanced Analytics/21-05/euro.csv")
df = df.dropna()
df['INR'] = df['INR'].astype(float)
df['USD']= df['USD'].astype(float)

correlation_coefficient=df['INR'].corr(df['USD'])
print(f'Correlation coefficient is {correlation_coefficient}')

plt.scatter(df['INR'],df['USD'])
plt.xlabel('Indian Rupee')
plt.ylabel('US Dollar')
plt.title(f'Scatter Plot (Correlation: { correlation_coefficient:.2f})')
plt.legend()
plt.show()


# Joint plt ( Similar to scatterplot)
import seaborn as sns
sns.jointplot(x='INR', y='USD', data=df, kind='scatter')
plt.show()