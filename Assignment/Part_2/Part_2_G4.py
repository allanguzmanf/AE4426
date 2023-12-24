# - Sarah Blanc - 5854830
# - Allan Guzmán Fallas - 5718619
# - Simon Scherders - 5878845

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re 
from geopy.distance import geodesic
from datetime import datetime, timedelta, time, date
import matplotlib.pyplot as plt

# Part 2 (IFAM model)
# group_number gn 
gn = '4'
# Define the inputs

# L: set of flights
flights_df = pd.read_excel('Group_'+gn+'_P2.xlsx', sheet_name='Flight')
#change data type of colum 'Flight no.' ro string
flights_df['Flight Number'] = flights_df['Flight Number'].astype(str)
flights_list = flights_df['Flight Number'].to_list() 

# P: set of passenger itineraries
paths = pd.read_excel('Group_'+gn+'_P2.xlsx', sheet_name='Itinerary').set_index('Itin No.')

# P_p: set of passenger itineraries with Recapture Rate from itinerary p
recapture_p = pd.read_excel('Group_'+gn+'_P2.xlsx', sheet_name='Recapture Rate').set_index(['From Itinerary','To Itinerary'])
recapture_p.rename(columns={'From Itinerary': 'p', 'To Itinerary': 'r'}, inplace=True)

# K: set of aircraft types
aircraft_df = pd.read_excel('Group_'+gn+'_P2.xlsx', sheet_name='Aircraft')
aircraft_df.rename(columns={'Type': 'AC Type'}, inplace=True)
aircraft_df.set_index('AC Type', inplace=True)
aircraft = aircraft_df.to_dict(orient='index')
ac_list = list(aircraft.keys())

# make a dictionary with the itinerary as the key and the rest as a sub-dictionary
paths['Leg 1'] = paths['Leg 1'].astype(str)
paths['Leg 2'] = paths['Leg 2'].astype(str)

paths = paths.to_dict(orient='index')
path_list = list(paths.keys())

# flights = flights_df.to_dict(orient='index')
recapture_p = recapture_p.to_dict(orient='index')

# Drop the inner dicts and just keep the values of the inner dict
for key in recapture_p:
    recapture_p[key] = recapture_p[key]['Recapture Rate']

# For all paths if 'Leg 1' and 'Leg 2' are numbers then create a list with both legs else, drop the keys from the list, and create a new key called 'Legs'
# else just change the name of the key 'Leg 1' to 'Legs'
for key in path_list:
    legs = []
    if paths[key]['Leg 1'] != '0' and paths[key]['Leg 2'] != '0':
        legs.append(paths[key]['Leg 1'])
        legs.append(paths[key]['Leg 2'])
        paths[key]['Legs'] = legs
    elif paths[key]['Leg 1'] != '0':
        legs.append(paths[key]['Leg 1'])
        paths[key]['Legs'] = legs
    del paths[key]['Leg 1']
    del paths[key]['Leg 2']

# Define path 999 with a fare of 0 and a demand of 0 
paths[999] = {'Legs': [], 'Demand': 0, 'Fare': 0}

for k in aircraft:
    aircraft[k]['TAT'] = timedelta(minutes=aircraft[k]['TAT'])

flights = flights_df.merge(aircraft_df.reset_index()[['AC Type']], how='cross')
# Get the cost from the column named after the AC Type
flights['Cost'] = flights.apply(lambda row: row[row['AC Type']], axis=1)
flights.drop(columns=ac_list, inplace=True)

# Get the capacity from the dictionary
flights['Capacity'] = flights.apply(lambda row: aircraft[row['AC Type']]['Seats'], axis=1)
# List of unique airports from Origin and Destination columns
airports = list(set(flights['ORG'].unique()).union(set(flights['DEST'].unique())))
# Misc date just for adding the TAT
misc_date = date(1,1,1)

# Drop rows with distance > range
flights['Arrival'] = flights.apply(lambda row: (datetime.combine(misc_date,row['Arrival']) + aircraft[row['AC Type']]['TAT']).time(), axis=1)
flights['Overnight'] = flights.apply(lambda row: row['Arrival'] < row['Departure'], axis=1)

# Make flights dictionary with main keys: AC Type, with a sub dictionary of flight numbers and each with flight details
flights_dict = {}
for i in flights_list:
    flights_dict[i] = flights[flights['Flight Number'] == i].set_index('AC Type').to_dict(orient='index')

flights = flights_dict
# Create an empty list to store the data
data = []
# Iterate over airports, aircraft types, and flights
for l in flights:
    for k in flights[l]:
            for n in airports:
                if flights[l][k]['ORG'] == n:
                    data.append([k, n, l, flights[l][k]['Departure'], 'Departure'])
                if flights[l][k]['DEST'] == n:
                    data.append([k, n, l, flights[l][k]['Arrival'], 'Arrival'])

# Create a dataframe with the data
events = pd.DataFrame(data, columns=['AC Type','Airport', 'Flight N', 'Time', 'D_A'])

# Add the TAT to the arrival times
events.sort_values(by=['AC Type', 'Airport', 'Time'], inplace=True)

# Reset the numbering of the events
events.reset_index(drop=True, inplace=True)
# For each airport and aircraft type i need to create a loop of ground arcs, each starting from the last event and ending at the next event, if there is no more events then the last ground arc is the overnight arc and it ends at the first event of the next day (first event of the next day is the first event of the same airport and aircraft type)

ground_arcs = pd.DataFrame(columns=['AC Type', 'Airport', 'Start Time', 'End Time'])

for k in ac_list:
    for n in airports:
        df = events[(events['AC Type'] == k) & (events['Airport'] == n)].sort_values(by=['Time'])
        for i in range(len(df)):
            if i == 0:
                ground_arcs = pd.concat([ground_arcs, pd.DataFrame({'AC Type': k, 'Airport': n, 'Start Time': [df.iloc[-1]['Time']], 'End Time': [df.iloc[i]['Time']]})], ignore_index=True)
            else:
                ground_arcs = pd.concat([ground_arcs, pd.DataFrame({'AC Type': k, 'Airport': n, 'Start Time': [df.iloc[i-1]['Time']], 'End Time': [df.iloc[i]['Time']]})], ignore_index=True)

ground_arcs.sort_values(by=['AC Type', 'Airport'], inplace=True)

# Drop rows if start time and end time are the same
ground_arcs = ground_arcs[~ground_arcs.apply(lambda row: (row['Start Time'] == row['End Time']), axis=1)]
ground_arcs['Arc ID'] = ground_arcs.groupby(['AC Type', 'Airport']).cumcount()
ground_arcs['Arc ID'] = ground_arcs.apply(lambda row: str(str(row['Arc ID']) + '_' + str(row['Airport'])) , axis=1)
ground_arcs['Overnight'] = ground_arcs.apply(lambda row: (row['End Time'] < row['Start Time']), axis=1)

# Create a nodes_df
nodes_df = ground_arcs[['AC Type', 'Airport', 'Start Time']].rename(columns={'Start Time': 'Time'})

# Add a count number for each row group by AC Type and Airport
nodes_df['Node ID'] = nodes_df.groupby(['AC Type','Airport']).cumcount()

# make a dictionary with the ac type as main key and the airport as secondary key with the node as tertiary key and the time as value
nodes = {}
for k in ac_list:
    nodes[k] = {}
    for n in airports:
        nodes[k][n] = {}
        for i in nodes_df[(nodes_df['AC Type'] == k) & (nodes_df['Airport'] == n)]['Node ID']:
            nodes[k][n][i] = {'Time': nodes_df[(nodes_df['AC Type'] == k) & 
                                               (nodes_df['Airport'] == n) & 
                                               (nodes_df['Node ID'] == i)]['Time'].values[0]}
            
# Add node id to the events
events.merge(nodes_df, how='left', on=['AC Type', 'Airport', 'Time']).sort_values(by=['AC Type', 'Airport'])

# Add a dictionary call departures and another one arrivals to the nodes dictionary at k,n,i+1 with 
# the events that have the same ac type, airport and time as the node
for k in ac_list:
    for n in airports:
        for i in nodes_df[(nodes_df['AC Type'] == k) & (nodes_df['Airport'] == n)]['Node ID']:
            nodes[k][n][i]['Departures'] = list(events[(events['AC Type'] == k) & (events['Airport'] == n) & (events['Time'] == nodes[k][n][i]['Time']) & (events['D_A'] == 'Departure')]['Flight N'])
            nodes[k][n][i]['Arrivals'] = list(events[(events['AC Type'] == k) & (events['Airport'] == n) & (events['Time'] == nodes[k][n][i]['Time']) & (events['D_A'] == 'Arrival')]['Flight N'])

# n+: ground arcs originating at any node n (start time)
# n-: ground arcs ending at any node n (end time)
n_plus = ground_arcs[['Airport', 'AC Type', 'Start Time', 'Arc ID']].rename(columns={'Start Time': 'Time'})
n_minus = ground_arcs[['Airport', 'AC Type', 'End Time', 'Arc ID']].rename(columns={'End Time': 'Time'})

# Add a dictionary call n+ and another one n- to the nodes dictionary at k,n,i+1 with 
# the events that have the same ac type, airport and time as the node
for k in ac_list:
    for n in airports:
        for i in nodes_df[(nodes_df['AC Type'] == k) & (nodes_df['Airport'] == n)]['Node ID']:
            nodes[k][n][i]['n+'] = list(n_plus[(n_plus['AC Type'] == k) & (n_plus['Airport'] == n) & (n_plus['Time'] == nodes[k][n][i]['Time'])]['Arc ID'])
            nodes[k][n][i]['n-'] = list(n_minus[(n_minus['AC Type'] == k) & (n_minus['Airport'] == n) & (n_minus['Time'] == nodes[k][n][i]['Time'])]['Arc ID'])

overnight_arcs = ground_arcs[ground_arcs['Overnight'] == True][['AC Type', 'Airport', 'Arc ID']]
overnight_flights = []
for l in flights:
    for k in flights[l]:
        if flights[l][k]['Overnight']:
            overnight_flights.append([k, l])

overnight_flights = pd.DataFrame(overnight_flights, columns=['AC Type', 'Flight no.'])

# s_ip: binary variable indicating whether flight i is in itinerary p
s_ip = {}
for i in flights_list:
    for p in paths:
        s_ip[i,999] = 0
        if i in paths[p]['Legs']:
            s_ip[i,p] = 1
        else:
            s_ip[i,p] = 0

# Q_i: unconstrained demand for flight i = sum s_ip * demand of itinerary p for p in P
Q_i = {}
for i in flights_list:
    Q_i[i] = 0
    for p in paths:
        Q_i[i] += s_ip[i,p] * paths[p]['Demand']
path_list = list(paths.keys())
flight_list = list(flights.keys())
# Add entries to P_p for path 0 with a Recapture Rate of 1
for p in paths:
    recapture_p[p,999] = 1
    recapture_p[999,p] = 0

ploting = False
if ploting:
    import random
    import plotly.graph_objects as go

    for k in ac_list:
        df = events[events['AC Type'] == k].sort_values(by=['Time'])
        fig = go.Figure()
        
        # make the figure taller to fit all airports
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
        )
        
        # Set marker color based on 'D_A' column
        marker_color = ['red' if d_a == 'Departure' else 'blue' for d_a in df['D_A']]
        
        # Add ground arcs
        ground_arcs_k = ground_arcs[ground_arcs['AC Type'] == k]
        for i, row in ground_arcs_k.iterrows():
            # Generate a random color
            random_color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
            fig.add_shape(
                type="line",
                x0=row['Start Time'],
                y0=row['Airport'],
                x1=row['End Time'],
                y1=row['Airport'],
                line=dict(color=random_color, width=2)
            )
        
        fig.add_trace(go.Scatter(
            x=df['Time'], 
            y=df['Airport'], 
            mode='markers+text',
            marker=dict(color=marker_color),
            hovertemplate= '<b>Flight no.</b>: ' + df['Flight N'] + '<br>' ))
        
        
        fig.update_layout(
            title="Ground Arcs for AC Type " + k,
            xaxis_title="Time",
            yaxis_title="Airport",
        )
        fig.show()

# for each path p, key-value pair in initial pairs with key p and value 999 in a set
initial_pairs = {}
for p in path_list:
    initial_pairs[p] = set()
    initial_pairs[p].add(999)

# Notation

L = flights
P = paths
P0 = initial_pairs
K = ac_list
G = nodes
N = airports
Q = Q_i
R = recapture_p
Gr = ground_arcs

def PMF_n_iters(
        n_iters, pairs, # required inputs
        pi = {}, sigma= {}, P=P, L=L, 
        R=R, iter=0, prev_model = {}): # optional inputs
    
    if iter == 0:
        print('Base model: all itineraries to ficticious (999)')
        C = {k: set(v) for k, v in pairs.items()} # current pairs

    if iter > 0:
        print('Iteration number: ', iter, '\n')
        C = {k: set(v) for k, v in pairs.items()} # current pairs
        tpr_prime = {}
        # tpr = (fare_p - sum (π_i) for i being each flight in path p) - bpr * (fare_r - sum (π_j) for j being each flight in path p)) - σ_p
        print('Negative slack pairs: ')
        for p,r in R.keys():
            t_prime_pr = ((P[p]['Fare'] - sum(pi[i] for i in P[p]['Legs'])) -
                            (R[(p,r)]) *
                            (P[r]['Fare'] - sum(pi[j] for j in P[r]['Legs'])) -
                            (sigma[p]))
            if t_prime_pr < -0.0001:
                tpr_prime[p,r] = t_prime_pr
                print(str(p)+' -> '+str(r)+': ', t_prime_pr)
                
        pairs = list(tpr_prime.keys())
        
        for n_p in pairs:
            C[n_p[0]].add(n_p[1])


    if len(pairs) == 0:
        print('No new pairs, optimal solution found in previous iteration')
        return C, pi, sigma, prev_model
    
    if len(pairs) > 0:
        if iter != 0:
            print('New pairs: ', pairs, '\n')

        # Define the model
        m_n = gp.Model('IFAM')

        ''' 
        Notation
        P = paths with info on O, D, Demand, Fare, Legs
        L = flights list with info on AC Type, ORG, DEST, Departure, Arrival, Overnight, Capacity, Cost
        P0 = initial_pairs with key-value pair in initial pairs with key p and value 999 in a set
        K = Types of aircraft
        G = nodes with info on Node ID, AC Type, Airport, Time, Departures, Arrivals, n+, n-
        N = airports list
        Q = Q_i = unconstrained demand for flight i = sum s_ip * demand of itinerary p for p in P
        R = recapture_p = set of passenger itineraries with Recapture Rate from itinerary p
        Gr = ground_arcs with info on AC Type, Airport, Start Time, End Time, Arc ID, Overnight
        C = current pairs
        '''

        # Decision variables from FAM
        # f[i,k] [RELAXED] binary 1 if flight arc i is assigned to aircraft type k, 0 otherwise
        f = {}
        # y_ak = [RELAXED] integer number of aircraft of type k on the ground arc a
        y = {}

        for i in L:
            for k in K:
                f[i, k] = m_n.addVar(vtype=GRB.CONTINUOUS, name='f_' + str(i) + '_' + str(k))

        for k in K:
            for a in list(ground_arcs[(ground_arcs['AC Type'] == k)]['Arc ID']):
                y[a, k] = m_n.addVar(vtype=GRB.CONTINUOUS,lb=0, name='y_' + str(a) + '_' + str(k))

        # Decision variables from PMF
        # t_pr: number of passengers that would like to fly on itinerary p and are reallocated to itinerary r
        t = {}

        for p in C:
            for r in C[p]:
                t[p,r] = m_n.addVar(vtype=GRB.CONTINUOUS,lb=0,name='t_'+str(p)+'_'+str(r))
        m_n.update()

        # Objective function part from the FAM
        of = gp.quicksum(
            L[i][k]['Cost'] * 
            f[i,k] 
            for i in L for k in K)

        # Objective function part from the PMF
        of +=  gp.quicksum((P[p]['Fare'] - R[(p,r)] * P[r]['Fare']) * t[p,r] 
                        for p in C for r in C[p])

        # Define the objective function
        m_n.setObjective(of, GRB.MINIMIZE)

        # Define the constraints
        # Constraint 1 [FAM]: 
        # Each flight is assigned to exactly one aircraft type
        for i in L:
            m_n.addConstr((gp.quicksum(f[i,k] for k in K) == 1), name='one_ac' + str(i))

        # Constraint 2 [FAM]: 
        # The number of AC arriving = AC departing, for each type at each node
        # y_n+_k + sum(f_i,k) = y_n-_k + sum(f_i,k)
        for k in K:
            for n in airports:
                for i in nodes[k][n]:
                    m_n.addConstr((y[nodes[k][n][i]['n+'][0], k] + gp.quicksum(f[w,k] for w in nodes[k][n][i]['Departures']) == 
                                y[nodes[k][n][i]['n-'][0], k] + gp.quicksum(f[w,k] for w in nodes[k][n][i]['Arrivals']) ),
                                name='balance_' + str(i) + '_' + str(k) + '_' + str(n))

        # Constraint 3 [FAM]: 
        # The number of overnight arcs + the number of overnight flights = the number of aircraft of each type 
        # using overnight_arcs and overnight_flights
        # sum(y_a,k) + sum(f_i,k) = number of aircraft of type k
        for k in K:
            m_n.addConstr((gp.quicksum(y[a, k] for a in list(overnight_arcs[(overnight_arcs['AC Type'] == k)]['Arc ID'])) + 
                        gp.quicksum(f[i, k] for i in list(overnight_flights[(overnight_flights['AC Type'] == k)]['Flight no.'])) <= 
                        aircraft[k]['Units']), name='overnight_' + str(k))

        # Constraint 4 [MIXED]: 
        # removed (from flight i in path p) - recaptured (for flight i in path p) ≥ demand spillage (for flight i) - capacity (for flight i) assinged to aircraft type k
        # sum seats_k * f_ik -sum s_ip * t_pr - sum sum s_ip * brp * t_rp >= ds_i for all i but for r = 0 
        m_n.addConstrs((
            gp.quicksum(aircraft[k]['Seats'] * f[i,k] for k in K) +
            gp.quicksum(s_ip[i,p] * t[p,r] for p in C for r in C[p]) - 
            gp.quicksum(s_ip[i,r] * R[(p,r)] * t[p,r] for p in C for r in C[p]) >= 
            Q[i] for i in L), name='π')

        # Constraint 5 [PMF]: sum t_pr <= Dp for all p
        for p in C:
            m_n.addConstr((
                gp.quicksum(t[p,r] for r in C[p]) <= P[p]['Demand']), name='σ[' + str(p) + ']')

        # Update the model
        m_n.update()
        # Optimize the model but dont print the output
        m_n.setParam('OutputFlag', 0)
        m_n.optimize()
        print('Objective value: %0.0f' % (m_n.objVal), '\n')
        #m_n.write('IFAM'+str(iter)+'.lp')

        # Print the total runtime of the model
        print('Total runtime: %0.2f' % (m_n.Runtime), '\n')

        # Print the first 5 t decision variables
        print('Non-Null Decision variables:')
        it_t = 0
        it_f = 0
        for v in m_n.getVars():
            if v.X != 0 and v.VarName[0] == 't' and it_t < 5:
                print('%s = %g' % (v.VarName, v.X))
                it_t += 1
        print('\n')        
        
        for v in m_n.getVars():
            if v.X != 0 and v.VarName[0] == 'f' and it_f < 5:
                print('%s = %g' % (v.VarName, v.X))
                it_f += 1
        print('\n')
                

        print('Non-Null Dual variables:')
        it_pi= 0
        it_sigma = 0
        for c in m_n.getConstrs():
            if c.Pi != 0 and it_pi < 5 and c.constrName[0] == 'π':
                print('%s = %g' % (c.ConstrName, c.Pi))
                it_pi += 1
            if c.Pi != 0 and it_sigma < 5 and c.constrName[0] == 'σ':
                print('%s = %g' % (c.ConstrName, c.Pi))
                it_sigma += 1


        # Save dual variables in a dictionary
        pi_new = {}
        for c in m_n.getConstrs():
            if c.constrName[0] == 'π':
                flight_num_pi = c.ConstrName[2:-1]
                pi_new[flight_num_pi] = c.Pi

        sigma_new = {}
        for c in m_n.getConstrs():
            if c.constrName[0] == 'σ':
                path_num_sigma = int(re.findall(r'\d+', c.ConstrName)[0])    
                sigma_new[path_num_sigma] = c.Pi

        if iter == 0:
            print ('End of base model iteration\n')
        else:
            print('End of iteration number: ', iter, '\n')
        
        iter += 1

        if iter == n_iters:
            print('Max number of iterations reached')
            return C, pi_new, sigma_new, m_n
        else:
            return  PMF_n_iters(
                    n_iters,
                    C,
                    sigma = sigma_new, 
                    pi= pi_new,
                    iter=iter,
                    prev_model=m_n)
final_pairs, final_pi, final_sigma, modelito = PMF_n_iters(21, initial_pairs)




# FINAL MODEL

# Define the model
m = gp.Model('IFAM')

# Notation
# P = paths with info on O, D, Demand, Fare, Legs
# L = flights_list
# P0 = initial_pairs
# K = ac_list
# G = nodes
# N = airports
# Q = Q_i
# R = recapture_p


# Decision variables from FAM
# f[i,k] binary 1 if flight arc i is assigned to aircraft type k, 0 otherwise
f = {}
# y_ak = integer number of aircraft of type k on the ground arc a
y = {}

for i in L:
    for k in K:
        f[i, k] = m.addVar(vtype=GRB.BINARY, name='f_' + str(i) + '_' + str(k))

for k in K:
    for a in list(ground_arcs[(ground_arcs['AC Type'] == k)]['Arc ID']):
        y[a, k] = m.addVar(vtype=GRB.INTEGER,lb=0, name='y_' + str(a) + '_' + str(k))

# Decision variables from PMF
# t_pr: number of passengers that would like to fly on itinerary p and are reallocated to itinerary r
t = {}

for p in final_pairs:
    for r in final_pairs[p]:
        t[p,r] = m.addVar(vtype=GRB.INTEGER,lb=0,name='t_'+str(p)+'_'+str(r))
m.update()

# Objective function part from the FAM
of = gp.quicksum(flights[i][k]['Cost'] * f[i,k] for i in L for k in K) + \
     gp.quicksum((P[p]['Fare'] - R[(p,r)] * P[r]['Fare']) * t[p,r] for p in final_pairs for r in final_pairs[p])

# Define the objective function
m.setObjective(of, GRB.MINIMIZE)

# Define the constraints from the FAM
# Constraint 1 [FAM]: 
# Each flight is assigned to exactly one aircraft type
m.addConstrs((gp.quicksum(f[i,k] for k in K) == 1 for i in L), name='one_ac')

# Constraint 2 [FAM]: 
# The number of AC arriving (n+ and arrivals) = AC departing yn-, for each type at each node
# y_n+_k + sum(f_i,k) = y_n-_k + sum(f_i,k)
for k in K:
    for n in N:
        for i in G[k][n]:
            n_plus = y[G[k][n][i]['n+'][0], k]
            n_minus = y[G[k][n][i]['n-'][0], k]
            departures = gp.quicksum(f[w,k] for w in G[k][n][i]['Departures'])
            arrivals = gp.quicksum(f[w,k] for w in G[k][n][i]['Arrivals'])
            m.addConstr((n_plus + departures - n_minus - arrivals == 0),
                         name='balance_' + str(i) + '_' + str(k) + '_' + str(n))

# Constraint 3 [FAM]: 
# The number of overnight arcs + the number of overnight flights = the number of aircraft of each type 
# using overnight_arcs and overnight_flights
# sum(y_a,k) + sum(f_i,k) = number of aircraft of type k
for k in K:
    m.addConstr((gp.quicksum(y[a, k] for a in list(overnight_arcs[(overnight_arcs['AC Type'] == k)]['Arc ID'])) + 
                 gp.quicksum(f[i, k] for i in list(overnight_flights[(overnight_flights['AC Type'] == k)]['Flight no.'])) <= 
                 aircraft[k]['Units']), name='overnight_' + str(k))

# Constraint 4 [MIXED]: 
# Aircraft capacity constraint
# sum seats_k * f_ik -sum s_ip * t_pr - sum sum s_ip * brp * t_rp >= ds_i for all i but for r = 0 
m.addConstrs((gp.quicksum(aircraft[k]['Seats'] * f[i,k] for k in K) +
              gp.quicksum(s_ip[i,p] * t[p,r] for p in final_pairs for r in final_pairs[p]) - 
              gp.quicksum(s_ip[i,r] * R[(p,r)] * t[p,r] for p in final_pairs for r in final_pairs[p]) >= 
              Q[i] for i in L), name='π')

# Constraint 5 [PMF]: sum t_pr <= Dp for all p
for p in P:
    m.addConstr((
        gp.quicksum(t[p,r] for r in final_pairs[p]) <= P[p]['Demand']), name='σ[' + str(p) + ']')


print('\n','Final model', '\n')
# Update the model
m.update()
# Optimize the model but dont print the output
m.setParam('OutputFlag', 0)
m.optimize()
print('Objective value: %0.0f euros' % (m.objVal), '\n')

# Print the total runtime of the model
print('Total runtime: %0.2f seconds' % (m.Runtime),'\n')
# Print the first 5 t decision variables
print('Non-Null Decision variables:')
it_t = 0
it_f = 0
for v in m.getVars():
    if v.X != 0 and v.VarName[0] == 't' and it_t < 5:
        print('%s = %g' % (v.VarName, v.X))
        it_t += 1
        
for v in m.getVars():
    if v.X != 0 and v.VarName[0] == 'f' and it_f < 5:
        print('%s = %g' % (v.VarName, v.X))
        it_f += 1

print('\n')
events.sort_values(by=['Flight N'], inplace=True)
filtered_events = events[events.apply(lambda row: f[row['Flight N'],row['AC Type']].x == 1, axis=1)]
# pivot the time and airport columns
filtered_events = filtered_events.pivot(index=['Flight N','AC Type'], columns='D_A', values=['Airport', 'Time'])
# drop the top level of the multi index
filtered_events.columns = filtered_events.columns.droplevel(0)
filtered_events.reset_index(inplace=True)
filtered_events.columns = ['Fligt No.', 'AC Type', 'Airport Arrival', 'Airport Departure', 'Time Arrival', 'Time Departure']
filtered_events
#filtered_events.to_csv('filtered_events.csv', index=False)

# df with AC grounded at overnight on each airport
grounded_overnight = pd.DataFrame(columns=ac_list, index=airports)
overnight_arcs['count'] = overnight_arcs.apply(lambda row: round(y[row['Arc ID'], row['AC Type']].x, ndigits=0), axis=1)

# Reorganize the df, make the columns the unique AC types and the rows the unique airports and the values the count
for k in ac_list:
    grounded_overnight[k] = overnight_arcs[overnight_arcs['AC Type'] == k].groupby(['Airport']).sum()['count']

# Fill the NaN values with 0
grounded_overnight.fillna(0, inplace=True)
grounded_overnight['Total'] = grounded_overnight.sum(axis=1)
grounded_overnight = grounded_overnight[grounded_overnight['Total'] > 0].drop(columns='Total').astype(int)

print(grounded_overnight,'\n')#.to_excel('overnight.xlsx', sheet_name='grounded')

# Make a df with AC flying overnight and give the OD and the flight number
overnight_flights['count'] = overnight_flights.apply(lambda row: round(f[row['Flight no.'], row['AC Type']].x, ndigits=0), axis=1)
overnight_flights = overnight_flights[overnight_flights['count'] > 0].drop(columns='count')
overnight_flights['Origin'] = overnight_flights.apply(lambda row: flights[row['Flight no.']][row['AC Type']]['ORG'], axis=1)
overnight_flights['Destination'] = overnight_flights.apply(lambda row: flights[row['Flight no.']][row['AC Type']]['DEST'], axis=1)
overnight_flights['Departure'] = overnight_flights.apply(lambda row: flights[row['Flight no.']][row['AC Type']]['Departure'], axis=1)
overnight_flights['Arrival'] = overnight_flights.apply(lambda row: flights[row['Flight no.']][row['AC Type']]['Arrival'], axis=1)
overnight_flights.sort_values(by=['AC Type', 'Flight no.'],inplace=True)


print(overnight_flights,'\n')
#.to_excel('overnight.xlsx', sheet_name='flights')
final_recapture = []
for v in m.getVars():
    if v.x > 0.001 and v.varName[0] == 't' and v.varName[-3:] != '999':
        p_from = int(re.findall(r'\d+', v.varName)[0])
        p_to = int(re.findall(r'\d+', v.varName)[1])
        final_recapture.append([p_from, p_to, round(v.x, ndigits=0), R[(p_from,p_to)]])

final_recapture = pd.DataFrame(final_recapture, columns=['From', 'To', 'Recapture', 'Recapture Rate'])
final_recapture['Passengers'] = final_recapture.apply(lambda row: row['Recapture'] * row['Recapture Rate'], axis=1)
print(final_recapture,'\n')

total_spillage = 0
for v in m.getVars():
    if v.x > 0.001 and v.varName[0] == 't' and v.varName[-3:] == '999':
        total_spillage += v.x

print('Total spillage: %g' % total_spillage)

# Base case inputs dictionary
user_input = int(input("Do you want to run the full model? (1 = yes, 0 = no): "))

# Create an input for user to change the base case inputs
if user_input == 1:
    # Notation

    L = flights_list
    P = paths
    K = ac_list
    G = nodes
    N = airports
    Q = Q_i
    R = recapture_p

    R_test = {}
    # if a recapture rate exists for p,r then keep it else set it to 0
    for p in P:
        for r in P:
            if (p,r) in R.keys():
                R_test[p,r] = R[(p,r)]
            else:
                R_test[p,r] = 0

    # Define the model
    m_u = gp.Model('IFAM')

    # Notation
    # P = paths with info on O, D, Demand, Fare, Legs
    # L = flights_list
    # P = initial_pairs
    # K = ac_list
    # G = nodes
    # N = airports
    # Q = Q_i
    # R = recapture_p


    # Decision variables from FAM
    # f[i,k] binary 1 if flight arc i is assigned to aircraft type k, 0 otherwise
    f = {}
    # y_ak = integer number of aircraft of type k on the ground arc a
    y = {}

    for i in L:
        for k in K:
            f[i, k] = m_u.addVar(vtype=GRB.BINARY, name='f_' + str(i) + '_' + str(k))

    for k in K:
        for a in list(ground_arcs[(ground_arcs['AC Type'] == k)]['Arc ID']):
            y[a, k] = m_u.addVar(vtype=GRB.INTEGER,lb=0, name='y_' + str(a) + '_' + str(k))

    # Decision variables from PMF
    # t_pr: number of passengers that would like to fly on itinerary p and are reallocated to itinerary r
    t = {}

    for p in P:
        for r in P:
            t[p,r] = m_u.addVar(vtype=GRB.CONTINUOUS,lb=0,name='t_'+str(p)+'_'+str(r))
    m_u.update()

    # Objective function part from the FAM
    of = gp.quicksum(flights[i][k]['Cost'] * f[i,k] for i in L for k in K) + \
        gp.quicksum((P[p]['Fare'] - R_test[(p,r)] * P[r]['Fare']) * t[p,r] for p in P for r in P)

    # Define the objective function
    m_u.setObjective(of, GRB.MINIMIZE)

    # Define the constraints from the FAM
    # Constraint 1 [FAM]: 
    # Each flight is assigned to exactly one aircraft type
    m_u.addConstrs((gp.quicksum(f[i,k] for k in K) == 1 for i in L), name='one_ac')

    # Constraint 2 [FAM]: 
    # The number of AC arriving (n+ and arrivals) = AC departing yn-, for each type at each node
    # y_n+_k + sum(f_i,k) = y_n-_k + sum(f_i,k)
    for k in K:
        for n in N:
            for i in G[k][n]:
                n_plus = y[G[k][n][i]['n+'][0], k]
                n_minus = y[G[k][n][i]['n-'][0], k]
                departures = gp.quicksum(f[w,k] for w in G[k][n][i]['Departures'])
                arrivals = gp.quicksum(f[w,k] for w in G[k][n][i]['Arrivals'])
                m_u.addConstr((n_plus + departures - n_minus - arrivals == 0),
                            name='balance_' + str(i) + '_' + str(k) + '_' + str(n))

    # Constraint 3 [FAM]: 
    # The number of overnight arcs + the number of overnight flights = the number of aircraft of each type 
    # using overnight_arcs and overnight_flights
    # sum(y_a,k) + sum(f_i,k) = number of aircraft of type k
    for k in K:
        m_u.addConstr((gp.quicksum(y[a, k] for a in list(overnight_arcs[(overnight_arcs['AC Type'] == k)]['Arc ID'])) + 
                    gp.quicksum(f[i, k] for i in list(overnight_flights[(overnight_flights['AC Type'] == k)]['Flight no.'])) <= 
                    aircraft[k]['Units']), name='overnight_' + str(k))

    # Constraint 4 [MIXED]: 
    # Aircraft capacity constraint
    # sum seats_k * f_ik -sum s_ip * t_pr - sum sum s_ip * brp * t_rp >= ds_i for all i but for r = 0 
    m_u.addConstrs((gp.quicksum(aircraft[k]['Seats'] * f[i,k] for k in K) +
                gp.quicksum(s_ip[i,p] * t[p,r] for p in P for r in P) - 
                gp.quicksum(s_ip[i,p] * R_test[(r,p)] * t[r,p] for p in P for r in P) >= 
                Q[i] for i in L), name='π')

    # Constraint 5 [PMF]: sum t_pr <= Dp for all p
    for p in P:
        m_u.addConstr((
            gp.quicksum(t[p,r] for r in P) <= paths[p]['Demand']), name='σ[' + str(p) + ']')

    # Update the model
    m_u.update()
    # Optimize the model but dont print the output
    m_u.setParam('OutputFlag', 1)
    print('Running the full model')
    m_u.optimize()
    print('Objective value: %0.0f' % (m_u.objVal))