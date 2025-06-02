import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math

df= sns.load_dataset('flights')
print(df)

df['yearMonth'] = pd.to_datetime("01-" + df['month'].astype(str) + "-" + df['year'].astype(str))
df.set_index('yearMonth', inplace=True)  # e.g. yearMonth will contain 01-01-1956, 01-02-1956,....

airP = df[['passengers']].copy(deep=True)
print(airP)

#Visualize initial data
plt.figure(figsize=(10,5))
sns.lineplot(x=airP.index,y=airP['passengers'])
plt.title("Monthly passengers over time")
plt.show()

#Decompose to check trend ad seasonality
decomposition = seasonal_decompose(airP.passengers,period=12)
decomposition.plot()
plt.show()

#  Stationarity Check
def test_stationarity(dataFrame,var,window=12):
  dataFrame['rollMean'] = dataFrame[var].rolling(window=window).mean()
  dataFrame['rollStd'] = dataFrame[var].rolling(window=window).std()
  adf_result = adfuller(dataFrame[var])
  p_value = adf_result[1]

  print(f"ADF p_value: {p_value:.4f}")
  if p_value < 0.05:
    print("The Time Series is stationary (reject H0)")
  else:
    print(" The Time series is not stationary (don not reject the H0)")

  plt.figure(figsize=(10,5))
  sns.lineplot(x=dataFrame.index,y=dataFrame[var],label="Original")
  sns.lineplot(x=dataFrame.index,y=dataFrame['rollMean'],label="Rolling Mean")
  sns.lineplot(x=dataFrame.index,y=dataFrame['rollStd'],label="Rolling Std")
  plt.title("Rolling Statistics")
  plt.legend()
  plt.show()

# Test Stationarity
test_stationarity(airP,'passengers')


airP['shift'] = airP.passengers.shift(1)
airP['shiftDiff'] = airP['passengers'] - airP['shift']
print(airP.head(20))
test_stationarity(airP.dropna(),'shiftDiff')

#Since shiftdiff of 1 is not working, let us try with 2
airP['shift'] = airP.passengers.shift(2)
airP['shiftDiff'] = airP['passengers'] - airP['shift']
print(airP.head(20))
test_stationarity(airP.dropna(),'shiftDiff')

airP['firstDiff']= airP['passengers'].diff()
airP['Diff12']=airP['passengers'].diff(12)

#PACF and ACF plots
plot_pacf(airP['firstDiff'].dropna(),lags=20)
plt.show()

plot_acf(airP['firstDiff'].dropna(),lags=20)
plt.show()

