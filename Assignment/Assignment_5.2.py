'''
Perform  multiple linear regression on Australian_student_performanceData.csv 
Use coloumn High school gpa, Entrance exam score, Attendance Rate, hrs study per week, Library Usage, Hours of 
sleep per night, project assignment scores, midterm exam scores, to predict final exam scores, 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Step 1 :  Load the dataset
df =pd.read_csv("Assignment/Australian_Student_PerformanceData (ASPD24).csv")

print(df.columns)

# #Step 2:  Split the dataset into features and target variable
X = df[['High School GPA',  'Entrance Exam Score', 'Attendance Rate', 'Hours of Study per Week', 
        'Library Usage', 'Hours of Sleep per Night', 'Project/Assignment Scores', 'Midterm Exam Scores']]

y = df['Final Exam Scores']

# step : 3 : Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# step : 4 : Fit multiple Linear regression to training data
model = LinearRegression()
model.fit(X_train, y_train)

# step : 5 : Make predictions on the test data
y_pred = model.predict(X_test)

#print("-------------------------------------Actual Vs Predicted (Test Set)-------------------------------------")
#Create a dataframe to compare actual and predicted values
comparison_df = pd.DataFrame({'Actual Final Values': y_test, 'Predicted Final Values': y_pred})
print(comparison_df.head(10)) # Display first 10 rows of the comparison dataframe
print("-----------------------------------------------------------------------------------------------------------------\n")



# step : 6 : Make new predictions with custom input
print("---------------------------------------New Predictions---------------------------------------")
#Create a dataframe for new input values, ensuring colimn names, match training data features


new_data_for_Predicition = pd.DataFrame({
    'High School GPA': [3.5, 3.7, 3.9],  #Example values for High School GPA
    'Entrance Exam Score': [85, 90, 95],  #Example values
    'Attendance Rate': [95, 98, 99],  #Example values for attendanc
    'Hours of Study per Week': [20, 25, 30],  #Exampl
    'Library Usage': [5, 6, 7],  #Example values for library
    'Hours of Sleep per Night': [7, 8, 9],  #Example values for sleep
    'Project/Assignment Scores': [85, 90, 95],  #Example values
    'Midterm Exam Scores': [85, 90, 95]  #Example values
})


new_predicitions = model.predict(new_data_for_Predicition)

print("Data for Predictions: ", new_data_for_Predicition)
print("-------------------------------------------------------------------")
print("Predicted Final Values: ", new_predicitions)
print("--------------------------------------------------------------------")