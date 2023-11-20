import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Problem 3: Aircraft and network model

# Create param

flight_distance = {     # Distance [km]
    'PDL': {'PDL': 0, 'LIS': 1461, 'OPO': 1536, 'FNC': 975, 'YTO': 4545, 'BOS': 3888},
    'LIS': {'PDL': 1461, 'LIS': 0, 'OPO': 336, 'FNC': 973, 'YTO': 5790, 'BOS': 5177},
    'OPO': {'PDL': 1536, 'LIS': 336, 'OPO': 0, 'FNC': 1244, 'YTO': 5671, 'BOS': 5081},
    'FNC': {'PDL': 975, 'LIS': 973, 'OPO': 1244, 'FNC': 0, 'YTO': 5515, 'BOS': 4851},
    'YTO': {'PDL': 4545, 'LIS': 5790, 'OPO': 5671, 'FNC': 5515, 'YTO': 0, 'BOS': 691},
    'BOS': {'PDL': 3888, 'LIS': 5177, 'OPO': 5081, 'FNC': 4851, 'YTO': 691, 'BOS': 0}
}

airports = ['PDL', 'LIS', 'OPO', 'FNC', 'YTO', 'BOS']   # Set N of airports
aircraft_types = ['A310', 'A320']                       # Set K of aircraft types

# Make PDL the hub

g = {'PDL': 0, 'LIS': 1, 'OPO': 1, 'FNC': 1, 'YTO': 1, 'BOS': 1}    # g_k = 0 if a hub is located at airport k; g_k = 1 otherwise

# Create parameters
Yield = 0.16   # Revenue per RPK [€/RPK]
param = {  
    'A310': {
        'CASK': 0.12,       # Cost per seat per km [$/seat/km]
        's': 240,           # Seats [units]
        'LF': 0.80,         # Load factor [-]
        'sp': 900,          # Speed [km/h]
        'LTO': 20,          # Take-off and landing time [min]
        'BT': 14,           # Utilization time per aircraft [h/day]
        'AC': 2,            # Number of aircrafts [units]
        'Range': 9600       # Range [km]
    },
    'A320': {
        'CASK': 0.11,      
        's': 160,
        'LF': 0.80,
        'sp': 870,
        'LTO': 12,
        'BT': 12,
        'AC': 2,
        'Range': 5400
    }
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

m = gp.Model("Aircraft and network model")

# Create variables

w = m.addVars(airports, airports, vtype=GRB.INTEGER, name="w")  # w_ij: flow from airport i to airport j that transfers at the hub
x = m.addVars(airports, airports, vtype=GRB.INTEGER, name="x")  # x_ij: direct flow from airport i to airport j
z = m.addVars(aircraft_types, airports, airports, vtype=GRB.INTEGER, name="z")  # z_kij: number of flights from airport i to airport j with aircraft k

# Set objective to maximize profit

m.setObjective(gp.quicksum(Yield * flight_distance[i][j] * (x[i, j] + w[i, j])
                           - gp.quicksum(param[k]['CASK'] * param[k]['s'] * flight_distance[i][j] * z[k, i, j]for k in aircraft_types)
                           for i in airports for j in airports), GRB.MAXIMIZE)

# Add constraints

# Constraint 1: number of passengers from airport i to airport j

m.addConstrs((x[i, j] + w[i, j] <= demand[i][j] for i in airports for j in airports))

# Constraint 2: Transfer passengers are only if the hub is not the origin or destination

m.addConstrs((w[i, j] <= demand[i][j] * g[i] * g[j] for i in airports for j in airports), "c2")

# Constraint 3: capacity verification for each flight leg
m.addConstrs((x[i, j] +
              gp.quicksum(w[i, m] * (1 - g[j]) for m in airports) +
              gp.quicksum(w[m, j] * (1 - g[i]) for m in airports) <=
              gp.quicksum(z[k, i, j] * param[k]['s'] * param[k]['LF'] for k in aircraft_types)
              for i in airports for j in airports), "c3")

# Constraint 4: same departing and arriving aircrafts per airport

m.addConstrs((gp.quicksum(z[k, i, j] for k in aircraft_types) == gp.quicksum(z[k, j, i] for k in aircraft_types) for i in airports for j in airports), "c4")

# Constraint 5: block time verification for each flight leg

m.addConstrs((gp.quicksum((flight_distance[i][j] / param[k]['sp'] + param[k]['LTO'] / 60) * z[k, i, j] for i in airports for j in airports if i != j) <=
            (param[k]['BT'] * param[k]['AC'] * 7) for k in aircraft_types), "c5")

# Define matrix a[k, i, j] based on aircraft range
a = {}
for k in aircraft_types:
    for i in airports:
        for j in airports:
            a[k, i, j] = 10000 if flight_distance[i][j] <= param[k]['Range'] else 0

print(a)

# Constrain 6: z to range limits
m.addConstrs((z[k, i, j] <= a[k, i, j] for k in aircraft_types for i in airports for j in airports), "c7")

# Constraint 7: no self flights

m.addConstrs((z[k, i, j] == 0 for k in aircraft_types for i in airports for j in airports if i == j), "c8")

# update model

m.update()

# Solve the model

m.optimize()

# Print the solution as a pd.DataFrame for each aircraft type

solution_z = pd.DataFrame(columns=airports, index=airports)


for k in aircraft_types:
    for v in m.getVars():
        if v.varName[0] == 'z' and v.varName[2:6] == k:
            solution_z.loc[v.varName[7:10], v.varName[11:14]] = v.x
    print('Total flights for aircraft type %s \n' % k, solution_z, '\n')

# Print the solution as a pd.DataFrame for each unique airport pair

solution_x_w = pd.DataFrame(columns=airports, index=airports) # x_ij + w_ij

for v in m.getVars():
    if v.varName[0] == 'x':
        solution_x_w.loc[v.varName[2:5], v.varName[6:9]] = v.x
    elif v.varName[0] == 'w':
        solution_x_w.loc[v.varName[2:5], v.varName[6:9]] += v.x

print('Direct passengers + Transfer passengers \n', solution_x_w, '\n')

print('Max profit: %g euros' % m.ObjVal)

# Calculate total flown hours and total possible hours per aircraft type

total_flown_hours = {}
total_possible_hours = {}

for k in aircraft_types:
    total_flown_hours[k] = 0
    total_possible_hours[k] = param[k]['BT'] * param[k]['AC'] * 7
    for i in airports:
        for j in airports:
            if i != j:
                total_flown_hours[k] += (flight_distance[i][j] / param[k]['sp'] + param[k]['LTO'] / 60) * z[k, i, j].x
    print('Utilization percentaje for aircraft type %s: %g' % (k, total_flown_hours[k] / total_possible_hours[k] * 100))


# Get the coordinates of each airport

coordinates = {
    'PDL': {'x': 37.7412, 'y': -25.6979},
    'LIS': {'x': 38.7813, 'y': -9.13592},
    'OPO': {'x': 41.2371, 'y': -8.67055},
    'FNC': {'x': 32.6949, 'y': -16.7789},
    'YTO': {'x': 43.6777, 'y': -79.6248},
    'BOS': {'x': 42.3656, 'y': -71.0096}
}

# Plot the network using folium from the dataframes

import folium

# Create map
m = folium.Map(location=[40, -20], zoom_start=3)

# Add airports
for i in airports:
    folium.Marker([coordinates[i]['x'], coordinates[i]['y']], popup=i).add_to(m)

# Add routes from dataframe solution_x_w

for i in airports:
    for j in airports:
        if i != j:
            folium.PolyLine(locations=[[coordinates[i]['x'], coordinates[i]['y']], [coordinates[j]['x'], coordinates[j]['y']]],
                            color="blue",
                            weight=solution_x_w.loc[i, j] / 100,
                            opacity=0.5).add_to(m)
            
# Save map
