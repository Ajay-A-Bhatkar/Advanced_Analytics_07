#Linear Programming Problem to maxumize profits for manufacturing chairs and tables
#Import all classes of PuLP module
from pulp import *

#Create the problem variable to contain the problem data
model = LpProblem("Factory Production Problem", LpMaximize)

#Create variables for Chairs and Tables
#Parameters: name, Lower limit, Upper limit, Data type
chairs = LpVariable("Chairs", 0, None, LpInteger) #Number of Chairs to produce
tables = LpVariable("Tables", 0, None, LpInteger) #Number of Tables to produce

#Create maximize objective function (profit)
model += 300 * chairs + 1000 * tables, "Profit"

#Create constraints
model += 5 * chairs + 10 * tables <= 400, "Wood Constraint"  # Wood availablity
model += 2 * chairs + 6 * tables <= 300, "Time Constraint"  # Labor availability
model += chairs >=2  # Minimum number of chairs to produce

#The problem is solved using PuLP's choice of Solver
model.solve()

#Each of the variable is printed with its resolved optimum value
for v in model.variables():
    print(v.name, "=", v.varValue)  # Print variable name and its value
#Additional information
print("Total Profit: ", value(model.objective))    #print objective function value (Total Profit)