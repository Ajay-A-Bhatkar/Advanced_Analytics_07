#Scaling Example
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('27-05/tips.csv')
total_bill = df[['total_bill']]

#1. Standard Scaling
standard_scaler = StandardScaler()
df[['total_bill_standard_scaled']] = standard_scaler.fit_transform(total_bill)

#2. MinMax Scaling
minmax_scaler = MinMaxScaler()
df[['total_bill_minmax_scaled']] = minmax_scaler.fit_transform(total_bill)

pd.set_option('display.max_columns', None)  #Show al columns
print(df[['total_bill','total_bill_standard_scaled','total_bill_minmax_scaled']].head(10))
pd.reset_option('display.max_columns')

