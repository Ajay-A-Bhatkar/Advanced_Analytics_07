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

# Train and Test Split
train_size = int(len(airP) * 0.7)
train, test = airP.iloc[:train_size], airP.iloc[train_size:]

# Fit ARIMA model
model_arima = ARIMA(train['passengers'], order=(1, 1, 1))
model_arima_fit = model_arima.fit()
arima_pred = model_arima_fit.predict(start=len(train), end=len(airP)-1)

# Add ARIMA predictions to datframe
airP['arimaPred'] = np.nan
airP.iloc[train_size:, airP.columns.get_loc('arimaPred')] = arima_pred.values
print(airP.tail())

# To calculate P and Q for SARIMAX
airP['diff_combined'] = airP['passengers'].diff(2).diff(12)
 
plot_pacf(airP['diff_combined'].dropna(), lags=50)
plt.title('PACF -  seasonal differenced data')
plt.show()

plot_acf(airP['diff_combined'].dropna(), lags=50)
plt.title('ACF -  seasonal differenced data')
plt.show()

# Fit SARIMAX model
model_sarimax = SARIMAX(train['passengers'], order=(1, 2, 1), seasonal_order=(1, 2, 1, 12))
model_sarimax_fit = model_sarimax.fit()
sarimax_pred = model_sarimax_fit.predict(start=len(train), end=len(airP)-1)

# Add SARIMAX predictions to dataframe
airP['sarimaxPred'] = np.nan
airP.iloc[train_size:, airP.columns.get_loc('sarimaxPred')] = sarimax_pred.values
print(airP.tail(20))

# PLot predictions
plt.figure(figsize=(12, 6))
sns.lineplot(x=airP.index, y=airP['passengers'], label='Actual_passengers')
sns.lineplot(x=airP.index, y=airP['arimaPred'], label='ARIMA Predictions')
sns.lineplot(x=airP.index, y=airP['sarimaxPred'], label='SARIMAX Predictions')
plt.title('Actual Vs Predicted (ARIMA and SARIMAX)')
plt.legend()
plt.show()

# Future Forecasting using SARIMAX
future_dates = pd.DataFrame(pd.date_range(start='1961-01-01', end='1962-01-01', freq='MS'), columns=['Dates'])
future_dates.set_index('Dates', inplace=True)
print(future_dates)

future_forecast = model_sarimax_fit.get_forecast(steps=len(future_dates))
print(future_forecast)

#Plot future forecast
# plt.figure(figsize=(12, 6))
# sns.lineplot(x=airP.index, y=airP['passengers'], label='Actual_passengers')
# sns.lineplot(x=airP.index, y=airP['sarimaxPred'], label='SARIMAX Predictions')
# future_forecast.plot(color=-'black', label='Future Forecast')
# plt.title('SARIMAX Model with Future Forecast')
# plt.legend()
# plt.show()



# Arima Metrics

arima_mae= mean_absolute_error(airP['passengers'],airP['arimaPred'])
arima_mse = mean_squared_error(airP['passengers'],airP['arimaPred'])
arima_rmse= math.sqrt(arima_mse)
arima_r2 = r2_score(airP['passengers'],airP['arimaPred'])
print(f" ARIMA -> MAE: {arima_mae:.2f}, RMSE:{arima_rmse}, R^2:{arima_r2:.2f}")

# Sarimax Metrics
sarimax_mae= mean_absolute_error(airP['passengers'],airP['sarimaxPred'])
sarimax_mse= mean_squared_error(airP['passengers'],airP['sarimaxPred'])
sarimax_rmse= math.sqrt(sarimax_mse)
sarimax_r2 = r2_score(airP['passengers'],airP['sarimaxPred'])
print(f" SARIMAX -> MAE: {sarimax_pred_mse:.2f}, RMSE:{sarimax_rmse}, R^2:{sarimax_r2:.2f}")


import joblib
import pandas as pd
joblib.dump(model_sarimax_fit, 'Advanced_Analytics_07/02-06/sarimax_model.pkl')

#Let us say we need this model some time later
#Load the model
loaded_model = joblib.load('Advanced_Analytics_07/02-06/sarimax_model.pkl')

#Make new predictions
print("************ Making new predictions ************")
new_experience_years_for_predictions = [1.5, 4, 8, 12, 15, 20, 25]

for new_experience_year in new_experience_years_for_predictions:
    new_sal_prediction = loaded_model.predict(pd.DataFrame({'experience': [new_experience_year]}))
    print(f"Predicted salary for {new_experience_year} years of experience: ${new_sal_prediction[0]:.2f}")

#