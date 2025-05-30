#Encoding Example

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Load the dataset
df = pd.read_csv("27-05/tips.csv")
print("Original 'day' coloumn:\n", df['day'].head())

# ------ One-Hot Encoding ------
df_ohe = pd.get_dummies(df, columns=['day'], prefix='day')
print("One-Hot Encoding 'day' column:\n", df_ohe.filter(like='day').head())
print("-------------------------------------------------------------------------------")
print("One-Hot Encoding 'day' column:\n", df_ohe.filter(like='day').tail())

#Same thing but drop redundant column
df_ohe = pd.get_dummies(df, columns=['day'], prefix='day', drop_first=True)
print("\nOne-Hot Encoded 'day' with drop first:\n",df_ohe.filter(like='day').head())
print("-------------------------------------------------------------------------------")
print("\nOne-Hot Encoded 'day' with drop first:\n",df_ohe.filter(like='day').tail())


#---------------Label Encoding-----------

label_encoder = LabelEncoder()
df['day_label'] = label_encoder.fit_transform(df['day'])
print("\nLabel Encoding 'day' column:\n", df['day_label'].head())
print("-------------------------------------------------------------------------------")
print("\nLabel Encoding 'day' column:\n", df['day_label'].tail())
print("\n\n\n")

# --------------Frequency Encoding------------

day_freq = df['day'].value_counts(normalize=False)
df['day_freq'] = df['day'].map(day_freq)
print("\nFrequency Encoding 'day' column:\n", df['day_freq'].head())
print("-------------------------------------------------------------------------------")
print("\nFrequency Encoding 'day' column:\n", df['day_freq'].tail())
