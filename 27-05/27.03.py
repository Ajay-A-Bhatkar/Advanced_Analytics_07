# Multiple Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# step : 1 : Read the data from the CSV file
df = pd.read_csv('27-05/happyscore_income.csv')
X = df[['adjusted_satisfaction', 'avg_income', 'median_income', 'income_inequality']]
y = df['happyScore']

# step : 2 : Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# step : 3 : Fit multiple Linear regression to training data
model = LinearRegression()
model.fit(X_train, y_train)

#Print the model coefficients and intercept of the model
print("----------------------------------------Model coefficients and Intercepts----------------------------------------")
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("-----------------------------------------------------------------------------------------------------------------")

# step : 4 : Make predictions on the test data
y_pred = model.predict(X_test)

print("-------------------------------------Actual Vs Predicted (Test Set)-------------------------------------")
#Create a dataframe to compare actual and predicted values
comparison_df = pd.DataFrame({'Actual Final Values': y_test, 'Predicted Final Values': y_pred})
print(comparison_df.head(10)) # Display first 10 rows of the comparison dataframe
print("-----------------------------------------------------------------------------------------------------------------\n")

# step : 5 : Evaluate the model using metrics
print("---------------------------------------Model Evaluation Metrics---------------------------------------")
#Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), R-Squared (R2)
print("Mean Absolute Error (MAE): ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE): ", mean_squared_error(y_test, y_pred))
print("R-Squared (R2): ", r2_score(y_test, y_pred))

# step : 6 : Make new predictions with custom input
print("---------------------------------------New Predictions---------------------------------------")
#Create a dataframe for new input values, ensuring colimn names, match training data features

new_data_for_predictions = pd.DataFrame({
    'adjusted_satisfaction': [35, 42, 61],  #Example values for satisfaction
    'avg_income': [40000, 50000, 60000],  #Example values for average income
    'median_income': [45000, 55000, 65000],  #Example values for median income
    'income_inequality': [32, 43, 51]  #Example values for income inequality
})

new_predicitions = model.predict(new_data_for_predictions)

print("Data for Predictions: ", new_data_for_predictions)
print("-------------------------------------------------------------------")
print("Predicted Final Values: ", new_predicitions)
print("--------------------------------------------------------------------")

