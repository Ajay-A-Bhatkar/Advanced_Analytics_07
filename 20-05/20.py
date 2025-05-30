import numpy as np

#1. Initial product names  with dtype = object to allow None later
product_names = np.array(['soap', 'toothpaste', 'shampoo', 'milk', 'biscuits'], dtype=object)
# print(product_names)

#2. 5 Random float prices between 100 and 1000, rounded to 2 decimal places
prices = np.round(np.random.uniform(100, 1000, size=5), 2)
# print(prices)

#3. 5 Random integers quantities between 1 and 50
quantities = np.random.randint(1, 51, size=5)
# print(quantities)

#4. Add python list products to numpy arrays
extra_products = ['butter', 'bread']
product_names = np.append(product_names, extra_products)
# print(product_names)

#Accept their prices and quantities from user for item in extra_products
for item in extra_products:
    price = float(input(f"Enter the price for {item}: "))
    quantity = int(input(f"Enter the quantity for {item}: "))
    prices = np.append(prices, price)
    quantities = np.append(quantities, quantity)

# 5. Display all arrays
print("\n All Products:")
for name, price, quantity in zip(product_names, prices, quantities):
    print(f"{name:12} | Price: {price:7.2f} | Quantity: {quantity}")

#6. Delete third item from product_names and prices
product_names = np.delete(product_names, 2)
prices = np.delete(prices, 2)
quantities = np.delete(quantities, 2)
print("\n Updated Products:") 
for name, price, quantity in zip(product_names, prices, quantities):
    print(f"{name:12} | Price: {price:7.2f} | Quantity: {quantity}")

#7. Modify milk price by adding 5
soap_index = np.where(product_names == 'soap')[0][0]
if soap_index.size > 0:
    prices[soap_index] += 6
print(prices)

#8. Insert toothbrush after first two products
toothbrush_index = 2
product_names = np.insert(product_names, toothbrush_index, 'toothbrush')
price = float(input(f"Enter the price for {item}: "))
quantity = int(input(f"Enter the quantity for {item}: "))
prices = np.insert(prices, toothbrush_index, price)
print(product_names)

#9. Search for biscuit and display
seacrh_index = np.where(product_names == 'biscuits')[0]
if seacrh_index.size > 0:
    i = seacrh_index[0]
    print(f"\nFound 'biscuits' -> price : {prices[i]:.2f}, quantity: {quantities[i]}")
else:
    print("Biscuits not found")

#10. Multiply all prices by 1.10
prices = prices * 1.10

#11. Average and standard deviation
average = np.mean(prices)
std_dev_price = np.std(prices)
print(f"Average price: {average:.2f}")
print(f"Standard deviation of prices: {std_dev_price:.2f}")

#12. Set biscuit name to None and count nulls
product_names[np.where(product_names == 'biscuits')[0]] = None
#count how many True values are in the BOOLEAN array -i.e., how many elements None
null_count = np.count_nonzero(product_names == None)
print(f"Number of null values: {null_count}")

#Final Display of non-null products
print("\n Final NON-NULL Products:")
for name, price, quantity in zip(product_names, prices, quantities):
    if name is not None:
        print(f"{name:12} | Price: {price:7.2f} | Quantity: {quantity}")