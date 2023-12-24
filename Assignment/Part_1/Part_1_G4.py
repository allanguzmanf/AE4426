import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
import gurobipy as gp
from gurobipy import GRB

# Base case inputs dictionary
user_input = int(input("Do you want to use the base case inputs? (1 = yes, 0 = no)"))

base_input = {
    'BT' : 10,
    'LF' : 0.8,
    'Fuel_C' : 1.42,
    'H_econ' : 0.3,
    'Add_TAT': 0.5
}


# Create an input for user to change the base case inputs
if user_input == 1:
    parameters = base_input
else:
    #input new values
    bt_input = input("Enter the block time (BT) in hours or press enter to use 10")
    lf_input = input("Enter the load factor (LF) or press enter to use 0.8")
    fuel_input = input("Enter the fuel cost (Fuel_C) in €/gal or press enter to use 1.42")
    econ_input = input("Enter economic benefit (H_econ) in ratio or press enter to use 30%")
    tat_input = input("Enter the additional TAT (Add_TAT) in ratio or press enter to use 50%")
    
    alt_input = {
        'BT' : base_input['BT'] if bt_input == '' else int(bt_input),
        'LF' : base_input['LF'] if lf_input == '' else float(lf_input),
        'Fuel_C' : base_input['Fuel_C'] if fuel_input == '' else float(fuel_input),
        'H_econ' : base_input['H_econ'] if econ_input == '' else float(econ_input),
        'Add_TAT': base_input['Add_TAT'] if tat_input == '' else float(tat_input)
    }
    parameters = alt_input

# Read the data from the csv file
airport_info = pd.read_csv('Group_4_Airport_info.csv')

# We make a dictionary with the airport code as the key the rest of the columns as sub-dictionaries
airports = {}

for index, row in airport_info.iterrows():
    airports[row['ICAO Code']] = row.to_dict()

airport_list = list(airports.keys())
# Create a dictionary of all the airports and their coordinates
airport_coords = {}

for i in airport_list:
    airport_coords[i] = (airport_info[airport_info['ICAO Code'] == i]['Latitude (deg)'].values[0],
                       airport_info[airport_info['ICAO Code'] == i]['Longitude (deg)'].values[0])
# Read the data distance file
distance_info = pd.read_csv('Group_4_Distances.csv')
distance_info.rename(columns={'Unnamed: 0': 'Origin'}, inplace=True)

# Create a dictionary with the distance info
distance = {}
for index, row in distance_info.iterrows():
    distance[row['Origin']] = row.to_dict()

# Remove the origin column from the distance
for i in distance:
    distance[i].pop('Origin', None)

# Import aircraft info
aircraft_info = pd.read_csv('Aircraft_info.csv')

# Add a column for BT (Block Time) to the aircraft_info dataframe
aircraft_info['BT'] = parameters['BT']  # [hours]

# Create a dictionary with the aircraft types as keys and the rest of the columns as sub-dictionaries
aircrafts = {}
for index, row in aircraft_info.iterrows():
    aircrafts[row['AC_type']] = row.to_dict()

# Rename the main keys in the aircraft dictionary to a numeric value with format AC_#

# Create a list of the keys in the aircraft dictionary
keys = list(aircrafts.keys())

# Create a new dictionary with the new keys
new_keys = {}
for i in range(len(keys)):
    new_keys[keys[i]] = 'AC_' + str(i+1)

aircraft_types = list(new_keys.values())

# Create a new dictionary with the new keys
aircrafts_new = {}
for i in range(len(keys)):
    aircrafts_new['AC_' + str(i+1)] = aircrafts[keys[i]]

# Replace the keys in the aircraft dictionary with the new keys
aircrafts = aircrafts_new

# Special hub conditions
# TAT for flights to the hub are 50% longer than the normal TAT for each aircraft type

# Add TAT to hub to the aircraft dictionary
for i in aircraft_types:
    aircrafts[i]['TAT_hub'] = aircrafts[i]['TAT'] * parameters['Add_TAT']

# Import demand info
demand_info = pd.read_csv('Group_4_Demand.csv')
demand_info.rename(columns={'Unnamed: 0': 'Origin'}, inplace=True)

# Create a dictionary with the demand info
demand = {}
for index, row in demand_info.iterrows():
    demand[row['Origin']] = row.to_dict()

# Remove the origin column from the demand
for i in demand:
    demand[i].pop('Origin', None)
# Make LIRF the hub with a dictionary of all the airports with a 1 if it is the hub and 0 if it is not
hub = airport_list[0]

g = {}
for i in airport_list:
    if i == hub:
        g[i] = 0
    else:
        g[i] = 1
# Revenue parameters

# Load factor
lf = parameters['LF']

# Create Yield matrix dict from formula in RPK using distance matrix
# Formula: Yield = 5.9 ∙ dij^(−0.76) + 0.043

yield_matrix = {}
for i in airport_list:
    yield_matrix[i] = {}
    for j in airport_list:
        if i == j:
            yield_matrix[i][j] = 0
        else:
            yield_matrix[i][j] = 5.9 * (distance[i][j] ** (-0.76)) + 0.043


# Cost parameters
'''
All aircraft are leased, and therefore a leasing cost needs to be accounted for. 
The weekly leasing cost is a fixed amount depending on the type of aircraft

Fuel cost formula
CF_kij = CF_k ∙ f ∙ dij / 1.5
Where 
CF_kij = fuel cost for aircraft type k on route i-j [€]
CF_k = fuel cost for aircraft type k [galon/km]
f = fuel cost [€/galon]
dij = distance between airport i and j [km]

Time-based costs formula
CT_kij = CT_k ∙ dij / V_k 
Where
CT_kij = total time-based cost for aircraft type k on route i-j [€]
CT_k = total time-based cost for aircraft type k [€/h]
dij = distance between airport i and j [km]
V_k = cruise speed for aircraft type k [km/h]

Variable costs formula
Op_Cost_kij = CX_kij + CF_kij + CT_kij

Fixed leg costs
CX_k depends on the aircraft type and is a fixed cost per flight
'''
# Create a dictionary with the Op_Cost for each aircraft type and airport using the aircraft type as the key and the rest of the columns as sub-dictionaries
f  = parameters['Fuel_C'] 
Op_Cost = {}
for k in aircraft_types:
    Op_Cost[k] = {}
    for i in airport_list:
        Op_Cost[k][i] = {}
        for j in airport_list:
            Op_Cost[k][i][j] = aircrafts[k]['Operating_c'] + \
                                 aircrafts[k]['Fuel_c'] * f * distance[i][j] / 1.5 + \
                                 aircrafts[k]['Time_c'] * distance[i][j] / aircrafts[k]['Speed']
            # It should be noted that for flights departing or arriving at the hub airport the operating costs can be assumed to be 30% lower due to economies of scale
            if i == hub or j == hub:
                Op_Cost[k][i][j] = Op_Cost[k][i][j] * (1 - parameters['H_econ'])
# Create model

m = gp.Model("Aircraft and network model")
# Create variables

# w_ij: flow from airport i to airport j that transfers at the hub
w = m.addVars(airport_list, airport_list, vtype=GRB.INTEGER, name="w") 

# x_ij: direct from airport i to airport j that does not transfer at the hub
x = m.addVars(airport_list, airport_list, vtype=GRB.INTEGER, name="x")

# z_kij: number of flights from airport i to airport j with aircraft k
z = m.addVars(aircraft_types, airport_list, airport_list, vtype=GRB.INTEGER, name="z")

# y_k: number of aircraft of type k
y = m.addVars(aircraft_types, vtype=GRB.INTEGER, name="y")
    
# Create objective function to maximize profit
# Revenue
# Rev1:revenue from flights
revenue = gp.quicksum((yield_matrix[i][j] * distance[i][j] * (x[i, j] + 0.9 * w[i, j])) for i in airport_list for j in airport_list)

# Costs
# Cost1: Fixed weekly leasing cost for all aircraft types
fixed_cost = gp.quicksum((aircrafts[k]['Lease_c'] * y[k]) for k in aircraft_types)

# Cost2: Operational costs per flight from i to j
# Op_Cost_kij = CF_kij + CT_kij + CX_k (for all aircraft types k)
# Multiply Op_Cost_kij by the corresponding z_kij to get the total cost for all flights from i to j
Operation_cost = gp.quicksum((Op_Cost[k][i][j] * z[k, i, j]) for k in aircraft_types for i in airport_list for j in airport_list)


# Objective function
# Full objective function with revenue and cost
m.setObjective(revenue - fixed_cost - Operation_cost,GRB.MAXIMIZE)


m.update()   
# Add constraints

# Constraint 1: number of passengers from airport i to airport j
# x_ij + w_ij <= demand_ij (for all i and j)
m.addConstrs((x[i, j] + w[i, j] <= demand[i][j] for i in airport_list for j in airport_list), name="c1")

# Constraint 2: Transfer passengers are only if the hub is not the origin or destination
# w_ij <= demand_ij * g_i * g_j (for all i and j)
m.addConstrs((w[i, j] <= demand[i][j] * g[i] * g[j] for i in airport_list for j in airport_list), "c2")

# Constraint 3: capacity verification for each flight leg
# x_ij + sum(w_im * (1 - g_j) for all m) + sum(w_mj * (1 - g_i) for all m) <= sum(z_kij * s_k * LF for all k) (for all i and j)
m.addConstrs((x[i, j] + 
              gp.quicksum(w[i, m] * (1 - g[j]) for m in airport_list) + 
              gp.quicksum(w[m, j] * (1 - g[i]) for m in airport_list) <= 
              gp.quicksum(z[k, i, j] * aircrafts[k]['Seats'] * lf for k in aircraft_types) 
              for i in airport_list for j in airport_list), "c3")

# Constraint 4: same departing and arriving aircrafts per airport
# sum(z_kij) = sum(z_kji) (for all i and k)
m.addConstrs((gp.quicksum(z[k, i, j] for j in airport_list) == gp.quicksum(z[k, j, i] for j in airport_list) for i in airport_list for k in aircraft_types), "c4")


# Constraint 5: block time verification for each aircraft total
# we should add a TAT for only incoming to the hub of 50% of the normal TAT
# sum((dij / sp_k + TAT_k + (TAT_hub for all j != hub)) * z_kij for all i and j) <= BT_k * y_k (for all k)
m.addConstrs(((gp.quicksum((distance[i][j] / aircrafts[k]['Speed'] + aircrafts[k]['TAT'] / 60  +
                          (aircrafts[k]['TAT_hub'] * (1 - g[j]) / 60)) * 
                           z[k, i, j] for i in airport_list for j in airport_list) <= 
                           aircrafts[k]['BT'] * y[k] * 7) for k in aircraft_types), "c5")


# Define matrix a[k, i, j] based on aircraft range and runway length
a = {}
for k in aircraft_types:
    for i in airport_list:
        for j in airport_list:
            if (distance[i][j] <= aircrafts[k]['Range'] and \
                airports[j]['Runway (m)'] >= aircrafts[k]['Runway'] and \
                airports[i]['Runway (m)'] >= aircrafts[k]['Runway']) or i == j:
                a[k, i, j] = 100000 
            else:
                a[k, i, j] = 0

# Constraint 6: aircraft range verification
# z_kij <= a_kij (for all k, i, j)
m.addConstrs((z[k, i, j] <= a[k, i, j] for k in aircraft_types for i in airport_list for j in airport_list), "c6")

# Constraint 7: no self flights
# z_kij = 0 (for all k and i)
m.addConstrs((z[k, i, j] == 0 for k in aircraft_types for i in airport_list for j in airport_list if i == j), "c7")


# Update model
m.update()
# Optimize model
m.optimize()
# Total number of aircraft of each type and utilization hours as a % of the total block time of each aircraft type
total_flown_hours = {}
total_possible_hours = {}
total_flights = {}
total_costs = {}
total_seat_km= {}

aircraft_metrics = []
for k in aircraft_types:
    total_costs[k] = 0
    total_flown_hours[k] = 0
    total_possible_hours[k] = aircrafts[k]['BT'] * y[k].x * 7
    total_flights[k] = 0
    total_seat_km[k] = 0
    for i in airports:
        for j in airports:
            if i != j:
                total_flown_hours[k] += (distance[i][j] / aircrafts[k]['Speed']) * z[k, i, j].x
                total_flights[k] += z[k, i, j].x
                total_costs[k] += Op_Cost[k][i][j] * z[k, i, j].x
                total_seat_km[k] += aircrafts[k]['Seats'] * distance[i][j] * z[k, i, j].x
    total_costs[k] += aircrafts[k]['Lease_c'] * y[k].x

    aircraft_metrics.append({
        'Aircraft Type': aircrafts[k]['AC_type'],
        'Total Costs (M€)': total_costs[k] / 1000000,
        'Number of Aircraft': y[k].x,
        'Utilization (%)': total_flown_hours[k] / total_possible_hours[k] * 100,
        'Number of Flights': total_flights[k],
        'Total Seat km (M)': total_seat_km[k] / 1000000,
    })

aircraft_metrics_df = pd.DataFrame(aircraft_metrics).set_index('Aircraft Type')
print('\n')
print('Objective function value (K€): %g' % (m.objVal / 1000))

# Calculate total revenue
total_revenue = 0
for i in airport_list:
    for j in airport_list:
        if i != j:
            total_revenue += yield_matrix[i][j] * distance[i][j] * (x[i, j].x + 0.9 * w[i, j].x)
print('Total revenue (M€):  %0.2f' % (total_revenue / 1000000))

# Calculate total costs
total_costs = 0
for k in aircraft_types:
    total_costs += aircrafts[k]['Lease_c'] * y[k].x
    for i in airports:
        for j in airports:
            if i != j:
                total_costs += Op_Cost[k][i][j] * z[k, i, j].x
print('Total costs (M€): %0.2f' % (total_costs / 1000000))

# Calculate overall utilization %
print('Overall utilization: %0.2f ' % (sum(total_flown_hours.values()) / sum(total_possible_hours.values()) * 100))

print('\n')
print(aircraft_metrics_df)#.to_csv('aircraft_metrics.csv')

# Based on demand and cost, calculate the max profit per OD pair if all demand is met
demand_met = pd.DataFrame(columns=airport_list, index=airport_list)

for i in airport_list:
    for j in airport_list:
        demand_met.loc[i, j] = (x[i, j].x + w[i, j].x)

demand_met = demand_met.astype(float)   

# Create a dataframe with demand
demand_perc = pd.DataFrame(columns=airport_list, index=airport_list)

for i in airport_list:
    for j in airport_list:
        demand_perc.loc[i, j] = demand[i][j]

# Make the numbers floats
demand_perc = demand_perc.astype(float)

# Total demand met
total_demand_met = demand_met.sum().sum()

# Total demand
total_demand = demand_perc.sum().sum()

# Total trips from i to j by aircraft type k
solution_z_k = []

for k in aircraft_types:
    solution_z = pd.DataFrame(columns=airport_list, index=airport_list)
    for v in m.getVars():
        if v.varName[0] == 'z' and v.varName[2:6] == k:
            solution_z.loc[v.varName[7:11], v.varName[12:16]] = v.x
            # Add the aircraft type to the dataframe
    # print('Total flights for aircraft type: %s \n' % aircrafts[k]['AC_type'], solution_z, '\n')
    solution_z['Aircraft type'] = aircrafts[k]['AC_type']
    solution_z_k.append(solution_z)

# Make a dataframe from solution_z_k with a column for each aircraft type
solution_z_k = pd.concat(solution_z_k, axis=0)
# Total flow trips from i to j x_ij 
solution_x = pd.DataFrame(columns=airport_list, index=airport_list)
solution_w = pd.DataFrame(columns=airport_list, index=airport_list)

for v in m.getVars():
    if v.varName[0] == 'x':
        solution_x.loc[v.varName[2:6], v.varName[7:11]] = v.x
    elif v.varName[0] == 'w':
        solution_w.loc[v.varName[2:6], v.varName[7:11]] = v.x

# print('Total direct pax per OD \n', solution_x, '\n')
# print('Total transfered at hub per OD \n', solution_w, '\n')
# Export the solutions to a csv files with a long dataset format
solution_z_k.reset_index(inplace=True)
solution_z_k.rename(columns={'index': 'Origin'}, inplace=True)
solution_z_k.set_index(['Aircraft type', 'Origin'], inplace=True)

# Transform the solution_z_k dataframe to a long dataset format
solution_z_k_long = solution_z_k.stack().reset_index()
solution_z_k_long.columns = ['Aircraft type', 'Origin', 'Destination', 'Total_trips']

# Transform the solution_w dataframe to a long dataset format
solution_w_long = solution_w.stack().reset_index()
solution_w_long.columns = ['Origin', 'Destination', 'Total_pax']

# Transform the solution_x dataframe to a long dataset format
solution_x_long = solution_x.stack().reset_index()
solution_x_long.columns = ['Origin', 'Destination', 'Total_pax']

# # Export the solutions long dataframes to a csv file
# solution_z_k_long.to_csv('P1_Solutions/solution_trips.csv')
# solution_w_long.to_csv('P1_Solutions/solution_pax_transfers.csv')
# solution_x_long.to_csv('P1_Solutions/solution_pax.csv')
total_direct_pax = solution_x_long['Total_pax'].sum()
total_transfer_pax = solution_w_long['Total_pax'].sum()
total_pax = total_direct_pax + total_transfer_pax
transfer_ratio = total_transfer_pax / total_pax
print('\n')
print('Total direct pax: %g' % total_direct_pax)
print('Total transfer pax: %g' % total_transfer_pax)
print('Total pax: %g' % total_pax)
print('Transfer ratio: %0.2f' % transfer_ratio)
# Create a OD dataframe for the airport pairs with origin and destination coordinates and total flights

results = airport_info[['City Name','ICAO Code', 'Latitude (deg)', 'Longitude (deg)']].merge(
    airport_info[['City Name','ICAO Code', 'Latitude (deg)', 'Longitude (deg)']], how='cross', suffixes=('_origin', '_destination')
    )

hub_point = Point(airport_coords[hub][1], airport_coords[hub][0])

# Creata a point and a line for each airport pair
results['Origin'] = results.apply(lambda row: Point(row['Longitude (deg)_origin'], row['Latitude (deg)_origin']), axis=1)
results['Destination'] = results.apply(lambda row: Point(row['Longitude (deg)_destination'], row['Latitude (deg)_destination']), axis=1)
results['Direct_flights'] = results.apply(lambda row: LineString([row['Origin'], row['Destination']]), axis=1)
results['Transfer_flights'] = results.apply(lambda row: LineString([row['Origin'], hub_point, row['Destination']]), axis=1)

# Add the total flights per aircraft type to the results dataframe
for k in aircraft_types:
    results['Total_flights_' + k] = solution_z_k_long[solution_z_k_long['Aircraft type'] == aircrafts[k]['AC_type']]['Total_trips'].values

# Add the total pax to the results dataframe
results['Direct_pax'] = solution_x_long['Total_pax']
results['Transfering_pax'] = 0
results['OD_transf_pax'] = 0

results.set_index(['ICAO Code_origin', 'ICAO Code_destination'], inplace=True)

solution_w_long_filtered = solution_w_long[solution_w_long['Total_pax'] > 0]
# Add a new column with the transfer pax to the results dataframe 
for i, row in solution_w_long_filtered.iterrows():
    results.at[(row['Origin'], hub),'Transfering_pax'] += row['Total_pax']
    results.at[(hub, row['Destination']),'Transfering_pax'] += row['Total_pax']
    results.at[(row['Origin'], row['Destination']),'OD_transf_pax'] += row['Total_pax']

# Reset the index of the results dataframe
results.reset_index(inplace=True)

# Drop columns that are not needed like coordinates
results.drop(columns=['Latitude (deg)_origin', 'Longitude (deg)_origin', 'Latitude (deg)_destination', 'Longitude (deg)_destination'], inplace=True)

# Drop the rows with same origin and destination
results.drop(results[results['ICAO Code_origin'] == results['ICAO Code_destination']].index, inplace=True)

# Create a column with the OD pair string, but if the pair is duplicated, the order is inverted
results['OD_pair'] = np.where(results['ICAO Code_origin'] < results['ICAO Code_destination'], 
                               results['ICAO Code_origin'] + '_' + results['ICAO Code_destination'], 
                               results['ICAO Code_destination'] + '_' + results['ICAO Code_origin'])

# Calculate the total pax and flights for each leg
results['Total pax'] = results['Direct_pax'] + results['Transfering_pax']
results['Total flights'] = results['Total_flights_AC_1'] + results['Total_flights_AC_2'] + results['Total_flights_AC_3']

# Export the results dataframe to a csv file
# results.to_csv('P1_Solutions/results.csv')
# Calculate the profit by OD pair using the results dataframe
for i, row in results.iterrows():
    o = row['ICAO Code_origin']
    d = row['ICAO Code_destination']
    results.at[i, 'Revenue'] = yield_matrix[o][d] * distance[o][d] * (row['Total pax']) 
    results.at[i, 'Operating Costs'] = Op_Cost['AC_1'][o][d] * row['Total_flights_AC_1'] + \
                                       Op_Cost['AC_2'][o][d] * row['Total_flights_AC_2'] + \
                                       Op_Cost['AC_3'][o][d] * row['Total_flights_AC_3']

results['Operational Profit'] = results['Revenue'] - results['Operating Costs']


# Summarize the results by OD pair
results_OD = results[['OD_pair','Total_flights_AC_1','Total_flights_AC_2','Total_flights_AC_3','Direct_pax','Transfering_pax','Total pax','Total flights']].groupby(
    ['OD_pair']).sum()
results_OD.reset_index(inplace=True)

filtered_results = results.drop_duplicates(subset=['OD_pair'], keep='first')
filtered_results.reset_index(inplace=True)

# Merge the results_OD dataframe with filtered_results to get the coordinates of the origin and destination
results_OD = results_OD.merge(filtered_results[['City Name_origin', 'City Name_destination', 'Origin', 'Destination', 'Direct_flights', 'Transfer_flights', 'OD_pair']],
                                        how='left', on='OD_pair')

# Rename the columns City Name_origin and City Name_destination to City A and City B
results_OD.rename(columns={'City Name_origin': 'City A', 'City Name_destination': 'City B'}, inplace=True)
results_OD.sort_values(by=['Total flights'], ascending=False)

# Export the results dataframe to a csv file
#results_OD.to_csv('P1_Solutions/results_OD.csv')
results_for_report = results_OD[results_OD['Total flights'] > 1][['City A','City B','Total_flights_AC_1','Total_flights_AC_2','Total_flights_AC_3','Direct_pax','Transfering_pax','Total pax','Total flights']]
#results_for_report.to_csv('P1_Solutions/results_for_report.csv')