import gurobipy as gp
from gurobipy import GRB

# Create a new model

m = gp.Model("P2P_L3")

# Create param

airports = ['A1', 'A2', 'A3']
aircrafts = ['AC1', 'AC2']
flight_distance = {'A1': {'A1': 0, 'A2': 2236, 'A3': 3201},
                   'A2': {'A1': 2236, 'A2': 0, 'A3': 3500},
                   'A3': {'A1': 3201, 'A2': 3500, 'A3': 0}}    # Distance [km]

demand = {'A1': {'A1': 0, 'A2': 1000, 'A3': 200},
          'A2': {'A1': 1000, 'A2': 0, 'A3': 300},
          'A3': {'A1': 200, 'A2': 300, 'A3': 0}} # Demand [pax/week]

param = {'CASK': 0.12,         # Cost per seat per km [$/seat/km]
              'LF': 0.75,      # Load factor [-]
              's': 120,        # Seats [units]
              'sp': 870,       # Speed [km/h]
              'LTO': 20,       # Take-off and landing time [min]
              'BT': 10,        # Utilization time pero aircraft [h/day]
              'AC': 2}         # Number of aircrafts [units]

Yield = 0.15    # Revenue per RPK [€/RPK]

# We want to maximize the profit for 1 week

# Problem 1: determine the frequency in each flight leg assuming that you can
# operate 2 aircraft with the characteristics presented in the param dict
# and you expect a revenue of 0.18 €/RPK;

# Create variables

# x_ij: number of passengers from airport i to airport j

x = m.addVars(airports, airports, vtype=GRB.INTEGER, name="x")

# z_ij: number of flights from airport i to airport j

z = m.addVars(airports, airports, vtype=GRB.INTEGER, name="z")

# Create objective function

m.setObjective(gp.quicksum(Yield * flight_distance[i][j] * x[i, j] - param['CASK'] * flight_distance[i][j] * z[i, j] * param['s'] for i in airports for j in airports), GRB.MAXIMIZE)

# Create constraints

# Constraint 1: number of passengers from airport i to airport j

m.addConstrs((x[i, j] <= demand[i][j] for i in airports for j in airports), "c1")

# Constraint 2: passengers from airport i to airport j must be equal to the number of flights from airport i to airport j

m.addConstrs((x[i, j] <= param['s'] * z[i, j] * param['LF'] for i in airports for j in airports), "c2")

# Constraint 3: same departing and arriving aircrafts per airport

m.addConstrs((gp.quicksum(z[i, j] for j in airports) == gp.quicksum(z[j, i] for j in airports) for i in airports), "c3")

# Constraint 4: limits the number of hours that we can schedule the aircrafts taking into account the block time and LTO

m.addConstr((gp.quicksum((flight_distance[i][j] / param['sp'] + param['LTO'] / 60) * z[i, j] for i in airports for j in airports if i != j) <= param['BT']* param['AC'] * 7),"c4")

# Update model

m.update()

# Solve the model

m.optimize()

# Print the solution

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % m.ObjVal)