import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.arange(2010, 2021)
b = np.array([100, 120, 130, 160, 180, 200, 240, 260, 280, 300, 320])

#Stimulated data age vs bp
x = np.arange(20, 70, 5) #ages from 20 to 65 in 5 year intervals
y = 0.8 * x + 80 #BP values

#------------ Plot 1: Age vs. Blood Pressure ------------------------
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o', color='green', linestyle='-', label='BP vs. Age')
plt.title('Blood Pressure vs. Age')
plt.xlabel('Age(Years)')
plt.ylabel('Blood Pressure(mmHg)')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

#------------ Plot 2: Age vs. Blood Pressure ------------------------
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='s', color='blue', linestyle='--', label='Annual sales')
plt.title('Annual sales trend')
plt.xlabel('Year')
plt.ylabel('Sales (in $1000s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()



#combined plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
#Plot 1: BP v/s Age
axes[0].plot(x, y, color='green', marker='o', linestyle='-', label = 'BP vs Age')
axes[0].set_title('Blood Pressure vs. Age')
axes[0].set_xlabel('Age(Years)')
axes[0].set_ylabel('Blood Pressure(mmHg)')
axes[0].grid(True)
axes[0].legend()
#Plot 2: Yearwise Sales
axes[1].plot(x, y, color='blue', marker='s', linestyle='--', label = 'Annual sales')
axes[1].set_title('Annual sales trend')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Sales (in $1000s)')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
# plt.show()

#Another version
fig = plt.figure(figsize=(14, 5))

#Plot1: BP vs Age
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, y, color='green', marker='o', linestyle='-', label = 'BP vs Age')
ax1.set_title('Blood Pressure vs. Age')
ax1.set_xlabel('Age(Years)')
ax1.set_ylabel('Blood Pressure(mmHg)')
ax1.grid(True)
ax1.legend()

#Plot2: Yearwise Sales
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x, y, color='blue', marker='s', linestyle='--', label = 'Annual sales')
ax2.set_title('Annual sales trend')
ax2.set_xlabel('Year')
ax2.set_ylabel('Sales (in $1000s)')
ax2.grid(True)
ax2.legend()

# Adjust Spacing
plt.tight_layout()
plt.show()
