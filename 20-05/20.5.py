import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('monthly-cola-production-in-austr.csv')

#Keep only January 
df = df[df['Month'].str.endswith('-01')]

#Extract columns
month = df['Month']
cola = df['cola']

plt.figure(figsize=(10, 6))
plt.plot(month, cola, 
         color='red',
         linestyle = '--',
         marker = 'D',
         linewidth=1.5,
         label='cola')

plt.title('January Production of Cola in Australia')
plt.xlabel('Month')
plt.ylabel('Cola Production')

#Extras
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()