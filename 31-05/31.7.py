#Linear Programming Problem to maxumize profits for car manufacturing
#Import all classes of PuLP module
from pulp import *

#Create the problem variable to contain the problem data
model = LpProblem("Car manufacturing Problem", LpMaximize)

#Create variables for Chairs and b
#Parameters: name, Lower limit, Upper limit, Data type
a = LpVariable("Sandwiches", 0, None, LpInteger) #Number of Chairs to produce
b = LpVariable("Burgers", 0, None, LpInteger) #Number of b to produce
c = LpVariable("Pizzas", 0, None, LpInteger) #Number of b to produce


#Create maximize objective function (profit)
model += 25* a + 50* b + 75* c, "Profit"

#Create constraints
model += 0.2* a + 0.3* b + 0.5* c <= 20, "Bread Constraint"  # Bread availability
model += 0.1* a + 0.2* b + 0.3* c <= 15, "Vegetables Constraint"  # Vegetables availability
model += 10* a + 15* b + 20* c <= 600, "Cooking time Constraint"  # Cooking time availability

#The problem is solved using PuLP's choice of Solver
model.solve()

#Each of the variable is printed with its resolved optimum value
for v in model.variables():
    print(v.name, "=", v.varValue)  # Print variable name and its value
#Additional information
print("Total Profit: ", value(model.objective))    #print objective function value (Total Profit)
