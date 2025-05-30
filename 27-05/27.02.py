import pandas as pd
from sklearn.linear_model import LinearRegression

#Step 1: Importing the dataset
df = pd.read_csv('27-05/salary.csv')
X = df[['YearsExperience']]
Y = df['Salary']

#Step2: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Print X and y for the test set
print("-------- Test Set Data -------------")
print("X_test(Experience):\n", X_test)
print("\nY_test(Salary):\n", Y_test)
print("----------------------------------------")

#Step 3: Fit Simple Linear Regression to Training Data

model = LinearRegression()
model.fit(X_train, Y_train)

#Step 4: Make Prediction
Y_pred = model.predict(X_test)
print("Predicted Salary for Test Set:\n", Y_pred)

#Step 5: Visualizing the Training set results
import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Step 6: Visualizing the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, model.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Step 7 - Make New Prediction
new_X_values_df = pd.DataFrame({'YearsExperience':[1, 3, 5, 8, 12, 15, 20, 25]})
new_Y_values = model.predict(new_X_values_df)
print("Predicion for new experience values (1, 3, 5, 8, 12, 15, 20, 25): \n", new_Y_values)

#Step 8: Model Evaluation:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-Squared (R2):", r2)