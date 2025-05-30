# Pandas series example... 
import pandas as pd
import numpy as np

#1. Initial product names and zero prices as Series
product_names = pd.Series(['Soap', 'Toothpaste', 'Shampoo', 'Body cream', 'Face wash'])
product_prices = pd.Series([0] * len(product_names), dtype='float64')


#2 Assign Random Prices between 100 and 1000
product_prices = pd.Series(np.round(np.random.uniform(100, 1000, size=5) ,2))

#3 Add new products
extra_products = ['Bread', 'Butter']
extra_prices = []
for item in extra_products:
    p = float(input('Enter the price for {item}: '))
    extra_prices.append(p)
#Append new products and their prices to the existing series
product_names = pd.concat([product_names, pd.Series(extra_products)], ignore_index=True)
product_prices = pd.concat([product_prices, pd.Series(extra_prices)], ignore_index=True)

#4. Display all products
print("\n All Products: ")
for name, price in zip(product_names, product_prices):
    print(f"{name:12} | Price: {price:7.2f}")

#5. Delete 'Shampoo'
drop_index = product_names[product_names == 'shampoo'].index

product_names = product_names.drop(drop_index)
product_names = product_names.reset_index(drop=True)

product_prices = product_prices.drop(drop_index)
product_prices = product_prices.reset_index(drop=True)

#6. Add Rs 5 to price of 'Body cream'
body_cream_index = product_names[product_names == 'Body cream'].index
if not body_cream_index.empty:
    product_prices.loc[body_cream_index] += 5

#7. Insert 'Toothbrush' at position 2, so keep rows 0 and 1 as they are
p = float(input('Enter the price for toothbrush: '))
product_names = pd.concat([product_names.iloc[:2],pd.Series(['toothbrush']), 
                           product_names.iloc[2:]]).reset_index(drop =True)
product_prices = pd.concat([product_prices.iloc[:2],pd.Series([p]),
                           product_prices.iloc[2:]]).reset_index(drop =True)

#8. Search for face wash
fw_index = product_names[product_names == 'Face wash'].index
if not fw_index.empty:
    print(f"Face wash -> price : {product_prices[fw_index[0]]:.2f}")
else:
    print("Face wash not found in the list.")

#9. Multiply all prices by 1.10
product_prices = product_prices * 1.10

#10. Average and Standard Deviation
print(f"\n Average Price: {product_prices.mean():.2f}")
print(f"Standard Deviation: {product_prices.std():.2f}")

#11. Set 'toothbrush' name to None  and count nulls
product_names[product_names == 'toothbrush'] = None
null_count = product_names.isnull().sum()
print(f"Null count: {null_count}")  

# Final display
print("\n Final Display Non null values: ")
for name, price in zip(product_names, product_prices):
    print(f"{name:12} | Price: {price:7.2f}")
