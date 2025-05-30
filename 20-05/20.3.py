import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x = np.arange(0, 10)
y=2*x

#Line Plot
plt.plot(x,y)
plt.show()

#Editing more figure paramters
plt.plot(x,y)
plt.xlabel('X axis Title')
plt.ylabel('Y axis Title')
plt.title('Plot Title')
plt.xlim(0,6) #lowerlimit, upper limit
plt.ylim(0,12)
# plt.show()

#Exporting a plot to a file
plt.plot(x,y)
# plt.savefig('example.png')

#Same Code, OO (Object Oriented) Syntax
fig, ax = plt.subplots()  #Create a figure and a set of subplots
ax.plot(x,y) # Plot on th axes object
ax.set_xlabel('X axis Title')
ax.set_ylabel('Y axis Title')
# plt.show()

#Alternative to OO syntax
fig = plt.figure(figsize=(6, 4))

#Add a subplot manually (1 row, 1 column, 1st subplot)
ax = fig.add_subplot(3,3,1)
ax = fig.add_subplot(3,3,2)
ax = fig.add_subplot(3,3,3)
ax = fig.add_subplot(3,3,4)
ax = fig.add_subplot(3,3,5)
ax = fig.add_subplot(3,3,6)
ax = fig.add_subplot(3,3,7)
ax = fig.add_subplot(3,3,8)
ax = fig.add_subplot(3,3,9)
ax.plot(x,y)
ax.set_xlabel('X axis Title')
ax.set_ylabel('Y axis Title')
ax.set_title('using subplot')
# plt.show()

"""
plt.plot(x,y)
"""