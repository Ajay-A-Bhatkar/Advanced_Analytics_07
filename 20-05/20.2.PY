import pandas as pd
import numpy as np

# 1. Initial products using DataFrame

product_data = {
    'Name': ['soap', 'toothpaste', 'shampoo', 'body cream', 'face wash'],
    'Price': [0] * 5
}

df = pd.DataFrame(product_data)
print("Initial Products: ")
print(df)

df['Price'] = pd.Series(np.round(np.random.uniform(100, 1000, size = 5), 2))
print("\nWith Random Prices: ")
print(df)


#3. Add New Products
extra_products = ['bread', 'butter']
new_df = pd.DataFrame({'Name': extra_products, 'Price': [0, 0]} )
df = pd.concat([df, new_df], ignore_index=True)

for item in extra_products:
    p = float(input(f"Enter the price of {item}: "))
    df.loc[df['Name'] == item, 'Price'] = p

#4 Display all products
print("\n All Products:")
print(df)

#5. Delete 'Shampoo'
df = df[df['Name'] != 'shampoo']
#use any of the 3 methods below
#df = df.query('Name != "shampoo"')
#df = df.loc[df['Name'] != 'shampoo']
df = df.drop(df[df['Name'] == 'shampoo'].index)

# 6. Modify price of 'soap' by adding 5

df.loc[df['Name'] == 'soap', 'Price'] += 5
print(df)

# 7. Insert 'toothbrush' at index 2

toothbrush_price = float(input("Enter price for toothbrush: "))
toothbrush_row = pd.DataFrame({'Name': ['toothbrush'], 'Price': [toothbrush_price]})
df = pd.concat([df.iloc[:2], toothbrush_row, df.iloc[2:]]).reset_index(drop = True)



# 8. Search for 'face wash'

facewash = df[df['Name'] == 'face wash']

if not facewash.empty:
  print(f"\nFound 'face wash' -> Price: {facewash.iloc[0]['Price']:.2f}")
else:
  print("\n'face wash' not found")

#9. Increase all prices by 10%
df['Price'] = df['Price'] * 1.10

# 10. Average and standard deviation

print(f"\nAverage Price: {df['Price'].mean():.2f}")
print(f"\nStandard Deviation: {df['Price'].std():.2f}")

# 11. Set 'toothbrush' name to None and count nulls

df.loc[df['Name'] == 'toothbrush', 'Name'] = None
null_count = df['Name'].isnull().sum()
print(f"\nNumber of null product names: {null_count}")

# Final display of non-null entries
print("\nNon-null Products Only:")
print(df[df['Name'].notnull()])
