#Linear Programming Problem to maxumize profits for car manufacturing
#Import all classes of PuLP module
from pulp import *

#Create the problem variable to contain the problem data
model = LpProblem("Car manufacturing Problem", LpMaximize)

#Create variables for Chairs and b
#Parameters: name, Lower limit, Upper limit, Data type
a = LpVariable("CAR A", 0, None, LpInteger) #Number of Chairs to produce
b = LpVariable("T", 0, None, LpInteger) #Number of b to produce

#Create maximize objective function (profit)
model += 30000 * a + 45000 * b, "Profit"

#Create constraints
R = 30
E = 30
model += 3 * a + 4 * b <= R, "Wood Constraint"  # Robot availablity
model += 5 * a + 6 * b <= 2 * E, "Time Constraint"  # Engineer availability
model += 1.5 * a + 3 * b <= 21  # Detailer availability
#non negativity constraints
model += a >= 0  
model += b >= 0  

#The problem is solved using PuLP's choice of Solver
model.solve()

#Each of the variable is printed with its resolved optimum value
for v in model.variables():
    print(v.name, "=", v.varValue)  # Print variable name and its value
#Additional information
print("Total Profit: ", value(model.objective))    #print objective function value (Total Profit)
