'''
1. Perform simple linear regression on weight-height.csv dataset.
predict the weight using height
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Step1: Load the dataset
df = pd.read_csv('Assignment/weight-height.csv')
X = df[['Height']]
Y = df['Weight']

#Step2: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Print X and y for the test set
print("-------- Test Set Data -------------")
print("X_test(Height):\n", X_test)
print("\nY_test(Weight):\n", Y_test)
print("----------------------------------------")

#Step 3: Fit Simple Linear Regression to Training Data

model = LinearRegression()
model.fit(X_train, Y_train)

#Step 4: Make Prediction
Y_pred = model.predict(X_test)
print("Predicted Weight for Test Set:\n", Y_pred)

#Step 5 - Make New Prediction
new_X_values_df = pd.DataFrame({'Height':[55, 58.6, 86]})
new_Y_values = model.predict(new_X_values_df)
print("Predicion for new Weight values ([55, 58.6, 86): \n", new_Y_values)


#Step 6: Model Evaluation:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-Squared (R2):", r2)