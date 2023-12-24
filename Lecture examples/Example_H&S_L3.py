import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Problem 2: Hub and spoke network

# Create param

flight_distance = {     # Distance [km]
    'PDL': {'PDL': 0, 'LIS': 1461, 'OPO': 1536, 'FNC': 975, 'YTO': 4545, 'BOS': 3888},
    'LIS': {'PDL': 1461, 'LIS': 0, 'OPO': 336, 'FNC': 973, 'YTO': 5790, 'BOS': 5177},
    'OPO': {'PDL': 1536, 'LIS': 336, 'OPO': 0, 'FNC': 1244, 'YTO': 5671, 'BOS': 5081},
    'FNC': {'PDL': 975, 'LIS': 973, 'OPO': 1244, 'FNC': 0, 'YTO': 5515, 'BOS': 4851},
    'YTO': {'PDL': 4545, 'LIS': 5790, 'OPO': 5671, 'FNC': 5515, 'YTO': 0, 'BOS': 691},
    'BOS': {'PDL': 3888, 'LIS': 5177, 'OPO': 5081, 'FNC': 4851, 'YTO': 691, 'BOS': 0}
}

airports = ['PDL', 'LIS', 'OPO', 'FNC', 'YTO', 'BOS'] 

# Make PDL the hub

g = {'PDL': 0, 'LIS': 1, 'OPO': 1, 'FNC': 1, 'YTO': 1, 'BOS': 1}    # g_k = 0 if a hub is located at airport k; g_k = 1 otherwise

# Create parameters
Yield = 0.16   # Revenue per RPK [€/RPK]
param = {  
    'CASK': 0.12,       # Cost per seat per km [$/seat/km]
    's': 150,           # Seats [units]
    'LF': 0.80,         # Load factor [-]
    'sp': 890,          # Speed [km/h]
    'LTO': 20,          # Take-off and landing time [min]
    'BT': 13,           # Utilization time pero aircraft [h/day]
    'AC': 4             # Number of aircrafts [units]
}

# Create demand
demand = {    # Demand [pax/week]
    'PDL': {'PDL': 0, 'LIS': 2509, 'OPO': 1080, 'FNC': 558, 'YTO': 770, 'BOS': 713},
    'LIS': {'PDL': 2509, 'LIS': 0, 'OPO': 216, 'FNC': 112, 'YTO': 360, 'BOS': 333},
    'OPO': {'PDL': 1080, 'LIS': 216, 'OPO': 0, 'FNC': 78, 'YTO': 46, 'BOS': 43},
    'FNC': {'PDL': 558, 'LIS': 112, 'OPO': 78, 'FNC': 0, 'YTO': 32, 'BOS': 30},
    'YTO': {'PDL': 770, 'LIS': 360, 'OPO': 46, 'FNC': 32, 'YTO': 0, 'BOS': 70},
    'BOS': {'PDL': 713, 'LIS': 333, 'OPO': 43, 'FNC': 30, 'YTO': 70, 'BOS': 0}
}

# Create model
m = gp.Model("H&S_L3")

# Create variables
w = m.addVars(airports, airports, vtype=GRB.INTEGER, name="w")  # w_ij: flow from airport i to airport j that transfers at the hub
x = m.addVars(airports, airports, vtype=GRB.INTEGER, name="x")  # x_ij: direct flow from airport i to airport j
z = m.addVars(airports, airports, vtype=GRB.INTEGER, name="z")  # z_ij: number of flights from airport i to airport j

# Create objective function
m.setObjective(gp.quicksum(Yield * flight_distance[i][j] * (x[i, j] + w[i, j]) - param['CASK'] * flight_distance[i][j] * z[i, j] * param['s'] for i in airports for j in airports), GRB.MAXIMIZE)

# Create constraints

# Constraint 1: number of passengers from airport i to airport j
m.addConstrs((x[i, j] + w[i, j] <= demand[i][j] for i in airports for j in airports), "c1")

# Constraint 2: Transfer passengers are only if the hub is not the origin or destination
m.addConstrs((w[i, j] <= demand[i][j] * g[i] * g[j] for i in airports for j in airports), "c2")

# Constraint 3: capacity verification for each flight leg
m.addConstrs((x[i, j] + 
              gp.quicksum(w[i, m] * (1 - g[j]) for m in airports) +
              gp.quicksum(w[m, j] * (1 - g[i]) for m in airports) <=
              z[i, j] * param['s'] * param['LF'] for i in airports for j in airports), "c3")

# Constraint 4: same departing and arriving aircrafts per airport
m.addConstrs((gp.quicksum(z[i, j] for j in airports) == gp.quicksum(z[j, i] for j in airports) for i in airports), "c4")

# Constraint 5: limits the number of hours that we can schedule the aircrafts taking into account the block time and LTO
m.addConstr((gp.quicksum((flight_distance[i][j] / param['sp'] + param['LTO'] / 60) * z[i, j] for i in airports for j in airports if i != j) <=
            param['BT']* param['AC'] * 7),"c5")

# Update model
m.update()

# Solve the model
m.optimize()

# Print the solution as a pd.DataFrame

solution_z = pd.DataFrame(columns=airports, index=airports)
solution_x = pd.DataFrame(columns=airports, index=airports)
solution_w = pd.DataFrame(columns=airports, index=airports)

for v in m.getVars():
    if v.varName[0] == 'z':
        solution_z.loc[v.varName[2:5], v.varName[6:9]] = v.x
    elif v.varName[0] == 'x':
        solution_x.loc[v.varName[2:5], v.varName[6:9]] = v.x
    elif v.varName[0] == 'w':
        solution_w.loc[v.varName[2:5], v.varName[6:9]] = v.x

print('Total flights \n', solution_z, '\n')
print('Direct passengers \n', solution_x, '\n')
print('Transfer passengers \n', solution_w, '\n')

print('Max profit: %g' % m.ObjVal)

# Calculate total flown hours

total_flown_hours = 0
total_possible_flown_hours = param['BT'] * param['AC'] * 7

for i in airports:
    for j in airports:
        if i != j:
            total_flown_hours += (flight_distance[i][j] / param['sp'] + param['LTO'] / 60) * z[i, j].x

print('Total flown hours: %g' % total_flown_hours)
print('Utilization percentaje of the fleet: %g' % (total_flown_hours / total_possible_flown_hours * 100))