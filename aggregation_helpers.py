import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import geopandas
from shapely import wkt
from shapely.geometry import Point
import signal
from contextlib import contextmanager
import time
from scipy.stats import norm, nbinom
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB
import gurobipy_pandas as gppd
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Function: uniform_sampling

Returns a sample of points distributed uniformly within a rectangular 2D-area

Inputs:

bound_x (2-Tuple of floats, default: (0,1)) - Lower and upper bound for the x-coordinates of the sampled points;
bound_y (2-Tuple of floats, default: (0,1)) - Lower and upper bound for the y-coordinates of the sampled points;
n (Integer) - Sample size;

Outputs:

result (list of shapely Points) - A uniform sample of shapely Points from the specified rectangular area;
"""

def uniform_sampling(bound_x : tuple = (0,1), bound_y: tuple = (0,1), n : int = 1):
    x = np.random.uniform(bound_x[0], bound_x[1], n)
    y = np.random.uniform(bound_y[0], bound_y[1], n)
    
    result = [Point(a,b) for (a,b) in zip(x,y)]
    
    return result

class TimeoutException(Exception): pass

"""
Function: time_limit (source: https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call)

Allows to terminate the execution of a piece of code upon exceeding a specified time limit.

Inputs:

seconds (float) - The number of seconds after which the code execution should terminate. On windows systems, this function will not work: A value of 0 should be passed in this case, causing the code to run indefinetely.

Outputs:

None
"""

@contextmanager
def time_limit(seconds):
    if seconds > 0:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        yield
        
"""
Function: independent_normal_sampling

Returns a sample of points following a (truncated) joint normal distribution with 0 covariances within a rectangular 2D-area

Inputs:

mean (2-Tuple of floats, default: (0.5,0.5)) - Mean parameters for the x and y coordinates to be sampled;
stdev (2-Tuple of floats, default: (0.5,0.5)) - Standard deviation parameters for the x and y coordinates to be sampled;
bound_x (2-Tuple of floats, default: (0,1)) - Lower and upper bounds for the x-coordinates of the sampled points;
bound_y (2-Tuple of floats, default: (0,1)) - Lower and upper bounds for the y-coordinates of the sampled points;
n (Integer) - Sample size;
underflow_rate (Float) - Maximal probability that the returned sample contains too few coordinates. A lower underflow rate will reslt in a larger sample from the normal distribution to ensure that the truncated sample contains enough values;

Outputs:

result (list of shapely Points) - An independent (truncated) normal sample of shapely Points from the specified rectangular area;
"""

def independent_normal_sampling(mean = (0.5,0.5), stdev= (0.5,0.5), bound_x = (0,1), bound_y= (0,1), n=1, underflow_rate = 0.000001):
    perc_draws_within_bounds_x = 1-norm.cdf(-(mean[0]-bound_x[0])/stdev[0])-norm.cdf((mean[0]-bound_x[1])/stdev[0])
    perc_draws_within_bounds_y = 1-norm.cdf(-(mean[1]-bound_y[0])/stdev[1])-norm.cdf((mean[1]-bound_y[1])/stdev[1]) 
    
    min_trials_x = int(n+nbinom.ppf(1-underflow_rate/2, n,perc_draws_within_bounds_x))
    min_trials_y = int(n+nbinom.ppf(1-underflow_rate/2, n, perc_draws_within_bounds_y))
    
    x = np.random.normal(mean[0], stdev[0], min_trials_x) 
    y = np.random.normal(mean[1],stdev[1], min_trials_y)
    
    x_trunc = [p for p in x if bound_x[0]<p<bound_x[1]][:n] 
    y_trunc = [p for p in y if bound_y[0]<p<bound_y[1]][:n]
    
    result = [Point(a,b) for (a,b) in zip(x_trunc,y_trunc)]
    
    return result

"""
Function: generate_problem_instance

Returns the locations and capacities of facilities and hubs in a plane given a specific input consisting of distributions and parameters.  

Inputs:

n_sources (Integer, default: 1000) - Number of source nodes to generate for the problem instance;
n_hubs (Integer, default: 10) - Number of processing hubs to generate for the problem instance;
sampling_func_source_locs (String, default: 'uniform_sampling') - Name of the (pre-specified) sampling function to use for to generate the locations of the sources; 
sampling_func_hub_locs (String, default: 'uniform_sampling') - Name of the (pre-specified) sampling function to use for to generate the locations of the hubs;  
sampling_vars_source_locs (Dictionary, default: {}) - Arguments to pass on to the locational sampling function of the sources;  
sampling_vars_hub_locs (Dictionary, default: {}) - Arguments to pass on to the locational sampling function of the hubs; 
dist_source_vols (String, default: 'uniform') - Function from numpy.random that is used to sample the volumes to be transported from each source node;
dist_hub_caps (String, default: 'normal') - Function from numpy.random that is used to sample the treatment capacities of each processing hub;
dist_vars_source_vols (List, default: [0,2]) - Parameter values to be passed on to the source volume sampling function;
dist_vars_hub_caps (List, default: [1000,100]) - Parameter values to be passed on to the hub capacity sampling function;
dist_hub_cost (String, default: 'uniform') - Function from numpy.random that is used to sample the opening/operating costs of each processing hub;
dist_vars_hub_cost (List, default: [200,200]) - Parameter values to be passed on to the hub cost sampling function; 
filepath_config (String, default: '') - Path to config file were all aforementioned parameters are stored in a specific way. If specified, the parameters read from the config file will override the parameters passed on to this function; 
bound_x (2-Tuple of floats, default: (0,1)) - Lower and upper bounds for the x-coordinates of the sampled points;
bound_y (2-Tuple of floats, default: (0,1)) - Lower and upper bounds for the y-coordinates of the sampled points;

Outputs:

sources (geopandas GeoDataFrame) - Dataframe containing locations and volumes of the sampled source nodes;
hubs (geopandas GeoDataFrame) - Dataframe containing locations, capacities, and opening/operating costs of the sampled processing hubs;
"""

def generate_problem_instance(n_sources: int = 1000, 
                              n_hubs: int = 10, 
                              sampling_func_source_locs : str = 'uniform_sampling', 
                              sampling_func_hub_locs : str = 'uniform_sampling', 
                              sampling_vars_source_locs : dict = {}, 
                              sampling_vars_hub_locs : dict = {}, 
                              dist_source_vols : str = 'uniform', 
                              dist_hub_caps : str = 'normal', 
                              dist_vars_source_vols : list = [0,2], 
                              dist_vars_hub_caps : list = [1000,100], 
                              dist_hub_cost : str = 'uniform',
                              dist_vars_hub_cost : list = [200,200],
                              filepath_config: str = '', 
                              bound_x : tuple = (0,1), 
                              bound_y: tuple = (0,1)):
    if filepath_config:
        try:
            config = pd.read_csv(filepath_config, index_col = [0])
        except:
            print(f"Error loading config file from location {filepath_config}. Perhaps you made a typo in the path? Default/specified values will be used for generation instead.")
        else:    
            try:
                n_sources, n_hubs = config['n']
                sampling_func_source_locs, sampling_func_hub_locs = config['sampling_func']
                dist_source_vols, dist_hub_vols = config['dist_vol']
                _, dist_hub_cost = config['dist_cost']
                
                n_sampling_vars_source_locs, n_sampling_vars_hub_locs = config['sampling_func_nparams'] 
                n_vars_source_vols, n_vars_hub_caps = config['dist_vol_nvars']
                _, n_vars_hub_cost = config['dist_cost_nvars']
                
                sampling_vars_source_locs = {config.loc['sources',f"sampling_func_param{i+1}_name"] : [v for v in config.loc['sources',[c for c in config.columns if f"sampling_func_param{i+1}_val" in c]]] for i in range(config.loc['sources','sampling_func_nparams'])}
                sampling_vars_hub_locs = {config.loc['hubs',f"sampling_func_param{i+1}_name"] : [v for v in config.loc['hubs',[c for c in config.columns if f"sampling_func_param{i+1}_val" in c]]] for i in range(config.loc['hubs','sampling_func_nparams'])}

                dist_vars_source_vols = [config.loc['sources',f"dist_vol_var{i+1}"] for i in range(n_vars_source_vols)]
                dist_vars_hub_caps = [config.loc['hubs',f"dist_vol_var{i+1}"] for i in range(n_vars_hub_caps)]
                dist_vars_hub_cost = [config.loc['hubs',f"dist_cost_var{i+1}"] for i in range(n_vars_hub_cost)]
                
            except:
                print("Error parsing config file. Not all values could be read from config. Please make sure the config is in the right format.")
                
                return None

    
    try:
        sources = geopandas.GeoDataFrame(geometry = globals()[sampling_func_source_locs](**sampling_vars_source_locs, n=n_sources), index = [f"Demand_point{(i+1):0{f'{len(str(n_sources))}'}d}" for i in range(n_sources)])
        hubs = geopandas.GeoDataFrame(geometry = globals()[sampling_func_hub_locs](**sampling_vars_hub_locs, n=n_hubs), index = [f"Facility{(i+1):0{f'{len(str(n_hubs))}'}d}" for i in range(n_hubs)])
    except:
        print("Error while sampling locations. Please check if the sampling variables are properly defined.")
        return None
    
    try:
        sources['volume'] = getattr(np.random,dist_source_vols)(*dist_vars_source_vols,n_sources)
        hubs['capacity'] = getattr(np.random,dist_hub_caps)(*dist_vars_hub_caps,n_hubs)
        hubs['cost'] = getattr(np.random,dist_hub_cost)(*dist_vars_hub_cost,n_hubs)
    except:
        print("Error while sampling volumes/capacities/costs. Please check if the distributional variables are properly defined")
        return None
        
    return sources, hubs

"""
Function: write_instance_config

Writes a config file (.csv) for an instance, given specific sampling distributions and parameter values.

Inputs:

n_sources (Integer, default: 1000) - Number of source nodes to generate for the problem instance;
n_hubs (Integer, default: 10) - Number of processing hubs to generate for the problem instance;
sampling_func_source_locs (String, default: 'uniform_sampling') - Name of the (pre-specified) sampling function to use for to generate the locations of the sources; 
sampling_func_hub_locs (String, default: 'uniform_sampling') - Name of the (pre-specified) sampling function to use for to generate the locations of the hubs;  
sampling_vars_source_locs (Dictionary, default: {}) - Arguments to pass on to the locational sampling function of the sources;  
sampling_vars_hub_locs (Dictionary, default: {}) - Arguments to pass on to the locational sampling function of the hubs; 
dist_source_vols (String, default: 'uniform') - Function from numpy.random that is used to sample the volumes to be transported from each source node;
dist_hub_caps (String, default: 'normal') - Function from numpy.random that is used to sample the treatment capacities of each processing hub;
dist_vars_source_vols (List, default: [0,2]) - Parameter values to be passed on to the source volume sampling function;
dist_vars_hub_caps (List, default: [1000,100]) - Parameter values to be passed on to the hub capacity sampling function;
dist_hub_cost (String, default: 'uniform') - Function from numpy.random that is used to sample the opening/operating costs of each processing hub;
dist_vars_hub_cost (List, default: [200,200]) - Parameter values to be passed on to the hub cost sampling function; 
filepath (String, default: '') - Path to config file (.csv) were all aforementioned parameters need to be stored;

Outputs:

config (pandas DataFrame) - Pandas Dataframe containing the config parameters;
"""

def write_instance_config(n_sources: int = 1000, 
                          n_hubs: int = 10, 
                          sampling_func_source_locs : str = 'uniform_sampling', 
                          sampling_func_hub_locs : str = 'uniform_sampling', 
                          sampling_vars_source_locs : dict = {}, 
                          sampling_vars_hub_locs : dict = {}, 
                          dist_source_vols : str = 'uniform', 
                          dist_hub_caps : str= 'normal', 
                          dist_vars_source_vols : list = [0,2], 
                          dist_vars_hub_caps : list = [1000,100],
                          dist_hub_cost : str = 'uniform',
                          dist_vars_hub_cost : list = [200,200],
                          filepath = 'config.csv'):
    ### Config basics
    config = pd.DataFrame({'sources':[n_sources,sampling_func_source_locs, len(sampling_vars_source_locs), dist_source_vols, len(dist_vars_source_vols), None, 0], 'hubs': [n_hubs,sampling_func_hub_locs, len(sampling_vars_hub_locs), dist_hub_caps, len(dist_vars_hub_caps), dist_hub_cost, len(dist_vars_hub_cost)]}, index= ['n','sampling_func', 'sampling_func_nparams', 'dist_vol', 'dist_vol_nvars', 'dist_cost', 'dist_cost_nvars'])
    
    ### Sampling parameters: Sources
    parameters_sampling_sources = pd.DataFrame(columns= ['sources'])
    
    for i, var in enumerate(sampling_vars_source_locs):
        parameters_sampling_sources.loc[f'sampling_func_param{i+1}_name'] = var
        try:
            for j, val in enumerate(sampling_vars_source_locs[var]):
                parameters_sampling_sources.loc[f'sampling_func_param{i+1}_val{j+1}'] = val
        except:
            parameters_sampling_sources.loc[f'sampling_func_param{i+1}_val1'] = sampling_vars_source_locs[var]

    ### Sampling parameters: Hubs
    parameters_sampling_hubs = pd.DataFrame(columns= ['hubs'])
    
    for i, var in enumerate(sampling_vars_hub_locs):
        parameters_sampling_hubs.loc[f'sampling_func_param{i+1}_name'] = var
        try:
            for j, val in enumerate(sampling_vars_hub_locs[var]):
                parameters_sampling_hubs.loc[f'sampling_func_param{i+1}_val{j+1}'] = val
        except:
            parameters_sampling_hubs.loc[f'sampling_func_param{i+1}_val1'] = sampling_vars_hub_locs[var]

    ### Volume distribution parameters: Sources
    parameters_distribution_sources = pd.DataFrame(columns = ['sources'])
    
    try:
        for i, var in enumerate(dist_vars_source_vols):
            parameters_distribution_sources.loc[f'dist_vol_var{i+1}'] = var
    except:
            parameters_distribution_sources.loc[f'dist_vol_var1'] = dist_vars_source_vols

    ### Capacity distribution parameters: Hubs            
    parameters_distribution_hubs = pd.DataFrame(columns = ['hubs'])
    
    try:
        for i, var in enumerate(dist_vars_hub_caps):
            parameters_distribution_hubs.loc[f'dist_vol_var{i+1}'] = var
    except:
            parameters_distribution_hubs.loc[f'dist_vol_var1'] = dist_vars_hub_caps
            
    ### Cost distribution parameters           
    parameters_cost = pd.DataFrame(columns = ['hubs'])
    
    try:
        for i, var in enumerate(dist_vars_hub_cost):
            parameters_cost.loc[f'dist_cost_var{i+1}'] = var
    except:
            parameters_cost.loc[f'dist_cost_var1'] = dist_vars_hub_cost
            
    ### Combine config parts and transposition
    parameters_sampling = pd.concat([parameters_sampling_sources, parameters_sampling_hubs], axis = 1).sort_index()
    parameters_distribution = pd.concat([parameters_distribution_sources, parameters_distribution_hubs], axis = 1).sort_index()
    
    config = pd.concat([config,parameters_sampling, parameters_distribution, parameters_cost]).T
    
    ### Write complete config to .csv
    config.to_csv(filepath)
    
    return config

"""
Function: plot_problem_instance

Visualizes the locations of the hubs and sources of an instance in the plane

Inputs:

sources (geopandas GeoDataFrame) - Dataframe containing locations and volumes of the source nodes for a particular instance;
hubs (geopandas GeoDataFrame) - Dataframe containing locations and capacities of the processing hubs for a particular instance;
savefile (String, default: '' ) - Path to file where the visualization should be stored;

Outputs:

None
"""

def plot_problem_instance(sources, hubs, savefile = ''):
    sources.plot(markersize=3*0.8*sources['volume']/sources['volume'].mean())
    hubs.plot(ax=plt.gca(),c='r', markersize = 250, marker = 's')
    
    plt.gcf().set_size_inches(10,10)
    
    if savefile:
        plt.savefig(savefile)
    plt.close()

"""
Function: get_euclidean_distances

Obtain the Euclidean distances between each point from a set of locations to each point from another set of locations.

Inputs:

location_set1 (geopandas GeoDataFrame) - GeoDataframe containing a 'geometry' column specifying a set of locations;
location_set2 (geopandas GeoDataFrame) - GeoDataframe containing a 'geometry' column specifying a set of locations;

Outputs:

result (geopandas GeoDataFrame) - A distance matrix specifying the distance between each pair of points from the location sets, where the locations in the first set are on the rows and the locations in the second set are on the columns of the matrix;
"""

def get_euclidean_distances(location_set1 : geopandas.GeoDataFrame, location_set2 : geopandas.GeoDataFrame):
    result = pd.DataFrame(cdist(location_set1.geometry.get_coordinates(), location_set2.geometry.get_coordinates()), index = location_set1.index, columns = location_set2.index)
    
    return result

"""
Function: standard_CFLP_formulation

Obtain an MILP formulation for a CFLP instance;

Inputs:

demand_points (geopandas GeoDataFrame) - GeoDataframe containing a 'geometry' column specifying a set of locations;
facilities (geopandas GeoDataFrame) - GeoDataframe containing a 'geometry' column specifying a set of locations;
distance_matrix (geopandas GeoDataFrame, default: empty geopandas GeoDataFrame) - A matrix-like object specifying the distances between the demand points and facilities;
valid_inequalities (boolean, default: True) - Whether to add valid inequalities to the model formulation;
problem_name (String, default: "Problem") - Name to pass on to the solver Model object;

Outputs:

prob (gurobipy Model) - Model object containing the MILP formulation of the problem instance corresponding to the specified demand points and facilities and their characteristics;
absolute_flows (pandas Series) - pandas Series containing linear solver expressions describing the absolute flows between demand points and facilities;
operation_vars (pandas Series) - pandas Series containing all operation variables of the model;
flow_vars (pandas Series) - pandas Series containing all 0-1 flow variables of the model;

"""

def standard_CFLP_formulation(demand_points : geopandas.GeoDataFrame, 
                              facilities : geopandas.GeoDataFrame, 
                              distance_matrix : geopandas.GeoDataFrame = geopandas.GeoDataFrame([]), 
                              valid_inequalities = True,
                              problem_name="Problem"):
    
    prob = gp.Model(problem_name)
    
    ### If no distance matrix specified, use Euclidean distances
    if len(distance_matrix) == 0:
        distance_matrix = get_euclidean_distances(facilities, demand_points)
    
    distance_table = pd.melt(distance_matrix, ignore_index=False, var_name='D').set_index('D', append=True)
    
    flow_vars = gppd.add_vars(prob, distance_table, name = "Flow", lb = 0, ub = 1, vtype = GRB.CONTINUOUS)
    operation_vars = gppd.add_vars(prob, facilities, name = "Operating", lb = 0, ub = 1, vtype = GRB.BINARY)
    absolute_flows = demand_points['volume'].mul(flow_vars, level = 1)
    
    prob.setObjective((distance_table['value']*absolute_flows).agg(gp.quicksum) + (facilities['cost']*operation_vars).agg(gp.quicksum), GRB.MINIMIZE)
    
    facility_inflow = absolute_flows.groupby(level = 0).agg(gp.quicksum) 
    demand_outflow = flow_vars.groupby(level = 1).agg(gp.quicksum) 
    
    ###Facility capacity constraints, demand flow conservation constraints
    gppd.add_constrs(prob, facility_inflow, GRB.LESS_EQUAL, operation_vars*facilities['capacity'])  
    gppd.add_constrs(prob, demand_outflow, GRB.EQUAL, 1)

    if valid_inequalities:
        m = gppd.add_constrs(prob, operation_vars.sub(flow_vars,level = 0), GRB.GREATER_EQUAL, 0)
        
    return prob, absolute_flows, operation_vars, flow_vars

"""
Function: cluster_kmeans

Returns a k-means clustering of a set of points (vectors) in a provided space.

Inputs:

space (ArrayLike) - A 2d-array containing a set of points (in any dimension) to cluster;
n_clusters (Integer) - The number of clusters the algorithm will provide;
weights (ArrayLike, default: None) - A vector of weights associated with the points to cluster;
n_init (Integer, default : 1) - The number of initializations with which the k-means ++ algorithm is run;

Outputs:

result (list of Strings) - List of strings specifying to which cluster each input point has been assigned;
"""
def cluster_kmeans(space: ArrayLike, n_clusters : int, weights: ArrayLike = None, n_init: int = 1):
    
    kmeans_obj = KMeans(n_clusters, n_init = n_init).fit(space, sample_weight = weights)
    cluster_vector = kmeans_obj.labels_
        
    result = [f"Cluster{i:0{len(str(n_clusters))}d}" for i in cluster_vector]
    
    return result

"""
Function: solve_simple_kmeans

Solve a CFLP instance using a simple planar k-means aggregation;

Inputs:

demand_points (geopandas GeoDataFrame) - GeoDataframe containing a 'geometry' column specifying a set of locations;
facilities (geopandas GeoDataFrame) - GeoDataframe containing a 'geometry' column specifying a set of locations;
distance_matrix (geopandas GeoDataFrame, default: empty geopandas GeoDataFrame) - A matrix-like object specifying the distances between the demand points and facilities;
n_clusters (Integer, default: 8) - The number of clusters to be passed on to the k-means++ algorithm, this is the number of ADP in the smaller MILP that will be solved;
n_init (Integer, default: 1) - The number of initializations with which the k-means ++ algorithm is run;
weighted (Boolean, default: True) - Indicates whether to weigh the demand points by their volume when clustering;
timeout (float, default: 1200) - Time limit in seconds which the compressed problem should not exceed;

Outputs:

result (Dictionary) - Dictionary containing the following keys:
                    > 'problem' : The allocation problem object
                    > 'flows' : Pandas Series containing the expressions for the flow solution of the allocation problem
                    > 'operation_vars' : Pandas Series containing the facility-operation variables
                    > 'clustering' : List specifying the clustering provided by the algorithm used to instantiate the smaller MILP
                    > 'compressed_problem' : The smaller MILP object
                    > 'time' : Runtime of the function in seconds 

"""

def solve_simple_kmeans(demand_points : geopandas.GeoDataFrame, 
              facilities : geopandas.GeoDataFrame, 
              distance_matrix : geopandas.GeoDataFrame = geopandas.GeoDataFrame([]), 
              n_clusters : int = 8,
              n_init : int = 1,
              weighted = True,
              timeout: float = 1200
              ):
    
    start_time = time.time()

    ###If weighted clustering applies, set weights equal to volume of demand points
    weights = demand_points['volume'] if weighted else None

    ### If no distance matrix specified, use Euclidean distances
    if len(distance_matrix) == 0:
        distance_matrix = get_euclidean_distances(facilities, demand_points)
    
    ### Perform clustering using Planar (Cartesian) coordinates
    clustering = cluster_kmeans(np.asarray([[x.coords[0][0],x.coords[0][1]] for x in demand_points.geometry]), n_clusters, weights, n_init = n_init)
    
    ### Compute the distance matrix associated with the aggregated/clustered demand points
    weighted_distance_sum = (distance_matrix*demand_points['volume']).groupby(clustering, axis=1).sum() 
    cluster_volumes = demand_points.groupby(clustering)[['volume']].sum()
         
    distance_matrix_clusters = weighted_distance_sum/cluster_volumes['volume'] 
    
    ### Initialize and solve smaller problem    
    compressed_prob, compressed_flows, compressed_operation_vars, _ = standard_CFLP_formulation(cluster_volumes, facilities, distance_matrix_clusters)
    if timeout > 0:
        compressed_prob.setParam('TimeLimit', timeout)
    compressed_prob.optimize()
    
    ### Initialize allocation problem
    prob, flows, operation_vars, _ = standard_CFLP_formulation(demand_points, facilities, distance_matrix, False)
    
    ### Fix operation variables to result from smaller problem
    gppd.add_constrs(prob, operation_vars, GRB.EQUAL, compressed_operation_vars.apply(lambda x: round((x+0).getValue())))
      
    ### Solve allocation problem        
    prob.optimize()

    result = {'problem': prob, 'flows': flows, 'operation_vars': operation_vars, 'clustering': clustering, 'compressed_problem': compressed_prob, 'time': time.time() - start_time} 
    
    return result  

"""
Function: bounding_procedure_intra_demand_distances

If no estimated distances are available, this function can be used to estimate the distances between demand points (at least between each demand point and a sample of the demand points). 
The function uses the triangle inequality to estimate the distances between demand points by a lower bound.

Inputs:

index_sample (List) - A list of indices of the demand points that are in the smaller reference sample;
distance_matrix (pandas DataFrame) - A distance matrix between demand points and facilities. The indices of the distance matrix should correspond to the indices of the demand points in the sample;

Outputs:

result (pandas DataFrame) - A DataFrame where the columns correspond to the indices of the demand points in the sample and the rows correspond to the indices of the demand points. The values in the DataFrame are the estimated distances between the demand points.
"""
def bounding_procedure_intra_demand_distances(index_sample, distance_matrix):
    ### For every point(index) in the sample, the distance to a demand point is at least equal to the largest difference between the distances of the sample point and demand point to any facility (Lower bound).
    ### An upper bound can also be determined, but this will be less close of an estimate in practice (smallest sum of sample and DP distance to a facility).
    result = pd.DataFrame({idx: distance_matrix.add(-distance_matrix.loc[idx], axis = 1).abs().max(axis = 1) for idx in index_sample})
    return result

"""
Function: phi

This function computes the sigmoidal decay function value for a given distance value, bandwidth, and alpha value.

Inputs: 

x (Float) - The distance value for which the decay function value should be computed;
bwx (Numpy ArrayLike) - A reference set of values of beta;
bw (Numpy ArrayLike) - A reference set of bandwidths corresponding to the values of beta specified in bwx;
bandwidth (Float, default: 0.3) - The bandwidth parameter of the decay function;
alpha (Float, default: 0.2) - The alpha parameter of the decay function;
gamma (Float, default: 1.0) - The gamma parameter of the decay function;

Outputs:

result (Float) - The decay function value for the given distance value and parameters;
"""	

def phi(x: float,bwx: ArrayLike, bw: ArrayLike, bandwidth: float = 0.3, alpha: float = 0.2, gamma: float = 1.0):
    beta = -np.interp(-bandwidth, -bw, -bwx)
    result = gamma*np.exp((x-alpha)*beta)/(1+np.exp((x-alpha)*beta))
    return result

"""
Function: obtain_alpha_reference

This function finds the reference points on the other side of the allocation boundary for a given facility. 
The reference point is the point in the sample with different allocation that is nearest in the direction of the straight line between the facility and the point.
This procedure is explained in detail in the paper in section 5.2 (figures 5a and 5b).

Inputs:

f - The facility with respect to which the reference points should be found;
facility_allocations (Pandas Series) - Series indicating to which facility the largest share of each demand point is allocated;
distance_matrix_T (Pandas DataFrame) - The distance matrix between demand points (rows) and facilities (colums), possibly augmented with surrogate points;
distance_matrix_intra_demand (Pandas DataFrame) - The distance matrix between demand points (rows) and a sample of demand points (columns), possibly augmented with surrogate points;
sample_demand (Pandas Index) - The sample of demand points to be used for reference point computation;
epsilon_local_inward (Float, default: 0.8) - The epsilon parameter for inward boundary estimation;
epsilon_local_outward (Float, default: 0.8) - The epsilon parameter for outward boundary estimation;
number_of_surrogates (Integer, default: 0) - The number of surrogate points that are used in the reference point computation;

Outputs:

result (Pandas Series) - The reference points for each demand point in the sample;
"""	

def obtain_alpha_reference(f,
                           facility_allocations: pd.Series,
                           distance_matrix_T: pd.DataFrame,
                           distance_matrix_intra_demand: pd.DataFrame,
                           sample_demand: pd.Index,
                           epsilon_local_inward: float = 0.8,
                           epsilon_local_outward: float = 0.8,
                           number_of_surrogates: int = 0):
    
    facility_non_allocations = facility_allocations != f
    sample_allocations = facility_allocations.loc[sample_demand]
    
    ### This is a reference array that has True values for the sample demand points that are not allocated to the facility and the surrogates;
    facility_non_allocations_ref = np.concatenate([sample_allocations != f, [True for i in range(number_of_surrogates)]])
    
    ### When no demand points are allocated to the facility, we need to make sure that the reference array contains at least one True point, where we will put the boundary.
    if sum(facility_non_allocations_ref) == len(sample_demand)+number_of_surrogates:
        closest_to_f = distance_matrix_T.loc[sample_demand, f].idxmin()
        sample_allocations.loc[closest_to_f] = f ### We make sure that the closest point to the facility is allocated to the facility, such that we can compute the boundary;

    ### This is a reference array that has True values for the sample demand points that are allocated to the facility;
    facility_allocations_ref = np.concatenate([sample_allocations == f, [False for i in range(number_of_surrogates)]])

    ### These are the relevant distance matrices for the reference point computation: From DPs allocated to f to the non-allocated sample points and from DPs not allocated to f to the allocated sample points;
    distance_matrix_allocated_non_allocated = distance_matrix_intra_demand.loc[facility_allocations == f, facility_non_allocations_ref]
    distance_matrix_non_allocated_allocated = distance_matrix_intra_demand.loc[facility_non_allocations, facility_allocations_ref]
    
    ### Apply the correction procedure to ensure that the reference points are more or less on a straight line between the facility and the demand point;
    distance_matrix_non_allocated_allocated = distance_matrix_non_allocated_allocated.add( epsilon_local_inward*distance_matrix_T.loc[distance_matrix_non_allocated_allocated.columns, f].values, axis = 'columns')
    distance_matrix_allocated_non_allocated = distance_matrix_allocated_non_allocated.add( -epsilon_local_outward*distance_matrix_T.loc[distance_matrix_allocated_non_allocated.columns, f].values, axis = 'columns')
    
    ### Combine the two Series of reference points;
    result = pd.concat([distance_matrix_non_allocated_allocated.idxmin(axis = 1),distance_matrix_allocated_non_allocated.idxmin(axis = 1)])

    return result
    
"""
Function: animate

Inputs:	
i (Integer) - The current iteration of the animation;
figure (Matplotlib pyplot Figure) - The figure that should be animated;
demand (geopandas GeoDataFrame) - GeoDataFrame containing the demand points;
facilities (geopandas GeoDataFrame) - GeoDataFrame containing the facilities;


Outputs:

result - The animation frame for iteration i;

"""
def animate(i: int,
            figure : plt.Figure,
            demand: geopandas.GeoDataFrame,
            facilities: geopandas.GeoDataFrame,
            clusterings: int, 
            operation_sols: int ,
            objective_values: int,
            iterations: int,
            open_facilities: int = None):
    
    global m1, m2
    
    #### Ensure previous plot is cleared
    figure.axes[0].clear()

    #### Make a series that contains the demand points and is indexed on geometry
    rp_dict = demand.copy().set_index('geometry')
    
    demand['cluster'] = clusterings[i+1]
    
    ### If a cluster column already exists, map clusters towards their 'peers' from previous frame
    try: 
        representative_points = demand.dissolve(by='cluster')['geometry'].representative_point()
        cluster_mappings = rp_dict.loc[representative_points]
        cluster_mappings.index = representative_points.index

        cluster_mappings.loc[cluster_mappings.duplicated('cluster'),'cluster'] = [c for c in demand['cluster'].unique() if c not in cluster_mappings['cluster'].unique()] 
        cm_dict = cluster_mappings['cluster'].to_dict()
        demand['cluster'] = demand['cluster'].apply(lambda x: cm_dict[x])
    
    except:
        pass

    ### Plot the demand points with the cluster colors on the left side of the animation
    cmap = mpl.colors.ListedColormap(['chocolate','gainsboro','slategray','palegoldenrod','blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    
    demand.plot(ax = figure.axes[0], column = 'cluster', cmap =  cmap, markersize = 3*0.8*demand['volume']/demand['volume'].mean())

    ### If the optimal solution is known, plot green squares as indicators
    try:
        open_facilities.plot(ax = figure.axes[0], c= 'g', marker = 's', markersize = 100*375/250)
    except:
        pass
    
    ### Plot the facilities and the current solution on the left side of the animation
    l = facilities.plot(ax = figure.axes[0], column = 'cap_cost_ratio', marker = 's', markersize = 100, cmap = 'Wistia')
    operation_sols[i+1].plot(ax = figure.axes[0], c= 'y', marker = 's', markersize = 100/(375/250)**2)
    
    ### Make sure that the old markers are not visible
    try:
        m1.set_visible(False)
        m2.set_visible(False)
        
    except:
        pass
        
    ### Add markers to the right side of the plot, which contains a chart with the objective progression over time
    m1 = figure.axes[1].axvline(i+1, c= '0.25', linestyle = ':', zorder = 50)
    m2 = figure.axes[1].scatter([i+1, i+1], objective_values[i+1], c = 'r', marker = 'D', zorder = 50)

    ### Set limits, size and title of the plot
    figure.axes[0].set_yticks([])
    figure.axes[0].set_xticks([])

    figure.axes[1].set_xlim(1,iterations)

    plt.gcf().set_size_inches(9.5,5)

    figure.suptitle(f'Algorithm iteration = {i+1}', size = 16)
    
    result = figure.axes[0].get_children() 
    
    return result

"""
Function: update_alpha

Updates the alpha values based on the current alpha values, the current iteration of the algorithm, and the current flows and operation variable values.

Inputs:
alpha (Pandas DataFrame) - The alpha values (indexed on DP and facility) to be updated;
t (Integer) - The current iteration of the algorithm;
alpha_reference (Pandas Series) - The reference points from the sample for each demand point;
distance_matrix_T_aug_elev (Pandas DataFrame) - The distance matrix between demand points and facilities, possibly augmented with surrogate points, which have an elevated distance value;
norm_factor (Float) - The normalization factor for the distance values;
eta (Float, default: 0.15) - The smoothing rate for the alpha update;

Outputs:
alpha_new (Pandas DataFrame) - The updated alpha values;

"""

def update_alpha(alpha: pd.DataFrame,
                 t: int,
                 alpha_reference: pd.Series,
                 distance_matrix_T_aug_elev: pd.DataFrame,
                 norm_factor: float,
                 eta: float = 0.15):            

    ### Determine alphas cprresponding to the reference points;            
    alpha_update_values_to_insert = alpha_reference.apply(lambda x: distance_matrix_T_aug_elev.loc[x,x.name].values/norm_factor, axis = 0).T
    
    ### Determine the alpha values to use for the update;
    alpha_update = alpha.copy()
    alpha_update.loc[alpha_update_values_to_insert.index, :] = alpha_update_values_to_insert 
        
    ### Update the alpha values; If the current alpha value is zero, we do not smooth;        
    alpha_new = (1-eta)*alpha + eta*alpha_update + alpha_update.mul((alpha.sum(axis=1) == 0).astype(int)*(1-eta), axis = 0 )

    return alpha_new

"""
Function: update_bandwidth

Updates the bandwidth values based on the current bandwidth values and the current iteration of the algorithm.

Inputs:

bandwidth (Pandas DataFrame) - The bandwidths (indexed on DP and facility) to be updated;
t (Integer) - The current iteration of the algorithm;
eta (Float, default: 0.15) - The learning rate for the bandwidth update;

Outputs:

updated_bandwidth (Pandas DataFrame) - The updated bandwidths;
"""
def update_bandwidth(bandwidth: pd.DataFrame, t: int, eta: float = 0.15):
    updated_bandwidth = bandwidth.apply(lambda x: np.maximum(x * (1-eta), 0.025))
    return updated_bandwidth

"""
Function: update_gamma

Updates the gamma values based on the current gamma values and the current iteration of the algorithm.

Inputs:

gamma (Pandas Series) - The gamma values (indexed on facility) to be updated;
t (Integer) - The current iteration of the algorithm;
best_solutions_list (Pandas DataFrame) - The list of best solutions found so far;
eta (Float, default: 0.18) - The learning rate for the gamma update;

Outputs:

gamma_new (Pandas Series) - The updated gamma values;
"""
def update_gamma(gamma: pd.Series, t: int, best_solutions_list: pd.DataFrame, eta: float = 0.18, warmup_duration = 6):
    best_solution_counts = best_solutions_list.iloc[:,:-1].sum()

    gamma_new = gamma*(1-eta)
    
    ### For facilities that are open in the best solutions, gamma will grow towards one at a rate that increases with the number of best solutions in which the facility is open;
    gamma_new += (best_solution_counts > 0) * ( (best_solution_counts-1)* -eta* gamma + best_solution_counts * eta)

    ### During the first iterations, gamma is kept constant and equal for all facilities
    gamma_new = gamma_new if t >= warmup_duration else  gamma
    
    return gamma_new 

def add_surrogate_points(demand,
                        facilities,
                        distance_matrix_intra_demand,
                        distance_matrix_T,
                        number_of_surrogates,
                        euclidean = True,
                        non_euclidean_inflation = 0.4):
    ### Obtain bounding box;
    x_min, y_min, x_max, y_max = demand.total_bounds 

    ### Generate surrogate points;
    x_surrogates = [Point(x_min, p) for p in np.linspace(y_min,y_max,int(number_of_surrogates/4))] + [Point(x_max, p) for p in np.linspace(y_min,y_max,int(number_of_surrogates/4))]  ### Vertical surrogate points;
    y_surrogates = [Point(p, y_min) for p in np.linspace(x_min,x_max,int(number_of_surrogates/4))] + [Point(p, y_max) for p in np.linspace(x_min,x_max,int(number_of_surrogates/4))]  ### Horizontal surrogate points;
                            
    surrogates = x_surrogates+y_surrogates ### All surrogate points;
    
    ### Store surrogate points in a Geopandas structure;
    surrogates_df = geopandas.GeoSeries(surrogates, index = [f"Surrogate_{i}" for i in range(len(surrogates))]) 

    ### Add the surrogate points to existing structures;
    distance_matrix_intra_demand_aug = pd.concat([distance_matrix_intra_demand,get_euclidean_distances(demand, surrogates_df)], axis=1) ### Add surrogates to the distance matrix columns (The DP->DP distance matrix);
    distance_matrix_T_aug = pd.concat([distance_matrix_T, (1+(1-euclidean)*non_euclidean_inflation)*get_euclidean_distances(surrogates_df, facilities)]) ### Add surrogates to the DM rows (The DP-> Facility distance matrix). Note the factor to compensate for using Euclidean estimates versus over the road distances in the original distance matrix;
    distance_matrix_T_aug_elev = pd.concat([distance_matrix_T, 2*(1+(1-euclidean)*non_euclidean_inflation)*get_euclidean_distances(surrogates_df, facilities)]) ### These are the distances to be used for alpha computation, note the multiplier to indicate that if a surrogate point is the nearest, the allocation boundary should lie very far (Outside the instance geometry); 

    return distance_matrix_intra_demand_aug, distance_matrix_T_aug, distance_matrix_T_aug_elev

def add_surrogate_points_unknown_coordinates(distance_matrix_intra_demand,
                                            distance_matrix_T,
                                            number_of_surrogates,
                                            padding_factor = 1.15):
     
    ### Estimate corner points: The first diagonal axis shall be the one spanning the largest distance between two individual demand points (At least, one demand point and one point in the sample); 
    cp1, cp2 = distance_matrix_intra_demand.stack().idxmax()
                    
    ### One of the points on the other diagonal axis should be the point that is furthest away from both endpoints of the first axis. We use a square root here to stimulate points that are roughly equally far away from both endpoints of the first axis;
    cp3 = (distance_matrix_intra_demand.loc[:,[cp1, cp2]]**(1/2)).sum(axis=1).idxmax()
    cp4 = pd.concat([(distance_matrix_intra_demand.loc[:,[cp1, cp2]])**(1/2),distance_matrix_intra_demand.loc[:,[cp3]]], axis = 1).sum(axis=1).idxmax()  ### For the other endpoint of the second axis, the same procedure but also make sure that it is far from the endpoint that is already known;
                    
    ### The distances of the surrogate points along the bounding box to any other point should be a weighted average of the distances from the respective corner points to that respective point; 
    surr_dists_13 = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_intra_demand.loc[:,cp1] + j*distance_matrix_intra_demand.loc[:,cp3]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(int(number_of_surrogates/4))])
    surr_dists_23 = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_intra_demand.loc[:,cp2] + j*distance_matrix_intra_demand.loc[:,cp3]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(int(number_of_surrogates/4),2*int(number_of_surrogates/4))])
    surr_dists_14 = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_intra_demand.loc[:,cp1] + j*distance_matrix_intra_demand.loc[:,cp4]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(2*int(number_of_surrogates/4),3*int(number_of_surrogates/4))])
    surr_dists_24 = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_intra_demand.loc[:,cp2] + j*distance_matrix_intra_demand.loc[:,cp4]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(3*int(number_of_surrogates/4),4*int(number_of_surrogates/4))])
                    
    surr_dists_13T = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_T.loc[cp1,:] + j*distance_matrix_T.loc[cp3,:]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(int(number_of_surrogates/4))])
    surr_dists_23T = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_T.loc[cp2,:] + j*distance_matrix_T.loc[cp3,:]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(int(number_of_surrogates/4),2*int(number_of_surrogates/4))])
    surr_dists_14T = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_T.loc[cp1,:] + j*distance_matrix_T.loc[cp4,:]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(2*int(number_of_surrogates/4),3*int(number_of_surrogates/4))])
    surr_dists_24T = pd.DataFrame([padding_factor * ((1 - j)*distance_matrix_T.loc[cp2,:] + j*distance_matrix_T.loc[cp4,:]) for j in np.linspace(0,1,int(number_of_surrogates/4))], index = [f"Surrogate_{i}" for i in range(3*int(number_of_surrogates/4),4*int(number_of_surrogates/4))])

    ### Add the surrogate points to existing structures
    distance_matrix_intra_demand_aug = pd.concat([distance_matrix_intra_demand, surr_dists_13.T, surr_dists_23.T, surr_dists_14.T, surr_dists_24.T], axis = 1)
    distance_matrix_T_aug = pd.concat([distance_matrix_T, surr_dists_13T, surr_dists_23T, surr_dists_14T, surr_dists_24T], axis = 0)
    distance_matrix_T_aug_elev = pd.concat([distance_matrix_T, 2*surr_dists_13T, 2*surr_dists_23T, 2*surr_dists_14T, 2*surr_dists_24T], axis = 0) ### These are the distances to be used for alpha computation

    return distance_matrix_intra_demand_aug, distance_matrix_T_aug, distance_matrix_T_aug_elev

def solve_dm_transformed_kmeans(demand_points : geopandas.GeoDataFrame, 
              facilities : geopandas.GeoDataFrame, 
              bwx: ArrayLike,
              bw: ArrayLike,
              distance_matrix : geopandas.GeoDataFrame = geopandas.GeoDataFrame([]), 
              n_clusters : int = 25,
              n_init = 1,
              bandwidth = 0.3,
              alpha = 0.2,
              gamma = 1.0,
              weighted = True,
              problem_name="Problem",
              problem_tuple = None,
              opt_operation_vars = None,
              solutions_seen = [],
              forced_exploration = False,
              max_forced_exploration = 1,
              iteration_timeout: float = 1200
              ):
    
    start_time = time.time()

    weights = demand_points['volume'] if weighted else None
    
    if len(distance_matrix) == 0:
        distance_matrix = get_euclidean_distances(facilities, demand_points)

    ### If alpha, bandwidth of gamma are not pandas structures, make them pandas structures with the same values for all demand points and facilities;
    if type(alpha) in [int,float]:
        alpha_ip = distance_matrix.copy()
        alpha_ip.loc[:,:] = alpha
        alpha = alpha_ip

    if type(bandwidth) in [int,float]:
        bandwidth_ip = distance_matrix.copy()
        bandwidth_ip.loc[:,:] = bandwidth
        bandwidth = bandwidth_ip

    if type(gamma) in [int,float]:
        gamma_ip = pd.Series(index = distance_matrix.index)
        gamma_ip.loc[:] = gamma
        gamma = gamma_ip

    print(f'{time.asctime()} - Decay function matrix computed')

    ###Weighted clustering using transformed normalized distances
    clustering = cluster_kmeans((distance_matrix/distance_matrix.max().max()).apply(lambda x: phi(x, bwx, bw, bandwidth.loc[x.name],alpha.loc[x.name], gamma[x.name]), axis = 1).T.values, n_clusters, weights, n_init = n_init)

    print(f'{time.asctime()} - Clustering obtained')

    weighted_distance_sum = (distance_matrix*demand_points['volume']).groupby(clustering, axis=1).sum() 
    cluster_volumes = demand_points.groupby(clustering)[['volume']].sum()
         
    distance_matrix_clusters = weighted_distance_sum/cluster_volumes['volume'] 
    
    compressed_prob, compressed_flows, compressed_operation_vars, _ = standard_CFLP_formulation(cluster_volumes, facilities, distance_matrix_clusters)
    if iteration_timeout > 0:
        compressed_prob.setParam('TimeLimit', iteration_timeout)
        
    if len(solutions_seen) > 0:
        gppd.add_constrs(compressed_prob, (compressed_operation_vars*solutions_seen.iloc[:,:-1]-compressed_operation_vars*(1-solutions_seen.iloc[:,:-1])).agg(gp.quicksum, axis=1), GRB.LESS_EQUAL, solutions_seen.iloc[:,:-1].agg(gp.quicksum, axis=1) - 1) ### Add constraints to ensure that already seen solutions are not seen again.
        facilities_never_seen = solutions_seen.iloc[:,:-1].sum(axis = 0) == 0

        print(f'{time.asctime()} - Facility exploration status:\n{len(facilities)-facilities_never_seen.sum()} ({(1-facilities_never_seen.sum()/len(facilities)):.1%}) seen - {facilities_never_seen.sum()} ({(facilities_never_seen.sum()/len(facilities)):.1%}) unseen out of {len(facilities)} total facilities')
        if forced_exploration:
    
            if facilities_never_seen.sum() >= 1: ### Only apply if some facilities have never been seen in a real solution
                print(f'{time.asctime()} - In this iteration, between 1 and {min(facilities_never_seen.sum(), max_forced_exploration)} unseen facilities will be explored')

                compressed_prob.addConstr((compressed_operation_vars*facilities_never_seen).agg(gp.quicksum), GRB.GREATER_EQUAL, min(facilities_never_seen.sum(), np.random.randint(max_forced_exploration)+1), name = 'force_exploration')

    print(f'{time.asctime()} - Aggregate problem set up')
    
    compressed_prob.optimize()
    
    if compressed_prob.status == GRB.OPTIMAL:
        print(f'{time.asctime()} - Aggregate problem solved')
    else:
        print(f'{time.asctime()} - Solving aggregate problem finished; Problem was not solved to optimality. Status code: {compressed_prob.status}')

    if type(opt_operation_vars) != type(None):
        compressed_prob_opt, compressed_flows_opt, compressed_operation_vars_opt, _ = standard_CFLP_formulation(cluster_volumes, facilities, distance_matrix_clusters, False)
        gppd.add_constrs(compressed_prob_opt, compressed_operation_vars_opt, GRB.EQUAL, opt_operation_vars)
        compressed_prob_opt.optimize()

        print(f'{time.asctime()} - Aggregate objective with optimal facility locations computed')
    if type(problem_tuple) == type(None):   
        prob, flows, operation_vars, _ = standard_CFLP_formulation(demand_points, facilities, distance_matrix, False)
    else:
        prob,flows,operation_vars = problem_tuple
        
    fixing_constraints = gppd.add_constrs(prob, operation_vars, GRB.EQUAL, compressed_operation_vars.gppd.X)

    print(f'{time.asctime()} - Allocation problem set up')
    
    prob.optimize()
    
    prob.remove(list(fixing_constraints))

    print(f'{time.asctime()} - Allocation problem optimized')
    
    result = {'problem': prob, 'flows': flows, 'operation_vars': operation_vars, 'clustering': clustering, 'compressed_flows':compressed_flows, 'compressed_problem': compressed_prob, 'compressed_dm': distance_matrix_clusters, 'alpha': alpha, 'comp_time': time.time()-start_time}

    if type(opt_operation_vars) != type(None):
        result['compressed_prob_opt'] =  compressed_prob_opt   

    return result 


def load_instance(set = 'CF', *args):
    if set == 'CF':
        return load_instance_CF(*args)
    elif set == 'Beasley':
        return load_instance_Beasley(*args)
    else:
        raise ValueError('Invalid instance set specified. Valid options are "CF", "Beasley".')

def load_instance_CF(i:int, prefix: str = ''):
    try:
        demand = pd.read_csv(f'Sources_Instance_{prefix}{i:04d}.csv', index_col = 0)
        facilities = pd.read_csv(f'Hubs_Instance_{prefix}{i:04d}.csv', index_col = 0)

        demand['geometry'] = demand['geometry'].apply(wkt.loads)
        demand = geopandas.GeoDataFrame(demand, geometry = 'geometry')
        facilities['geometry'] = facilities['geometry'].apply(wkt.loads)
        facilities = geopandas.GeoDataFrame(facilities, geometry = 'geometry')

        facilities['cap_cost_ratio'] = facilities['capacity']/facilities['cost']
                    
        print(f'{time.asctime()} - Instance {i} loaded from file')    
    except:
        raise ValueError(f'{time.asctime()} - Files "Sources_Instance_{prefix}{i:04d}.csv" and/or "Hubs_Instance_{prefix}{i:04d}.csv" not found.')
              
    try:
        distance_matrix = pd.read_csv(f"DM_{prefix}{i:04d}.csv", index_col = 0)
        distance_matrix.index = distance_matrix.index.astype(int)
        distance_matrix.columns = distance_matrix.columns.astype(int)
        euclidean = False
    except:
        print(f'{time.asctime()} - File "DM_{prefix}{i:04d}.csv" not found. Using Euclidean Distances.')
        distance_matrix = get_euclidean_distances(facilities, demand)
        euclidean = True
        
    return demand, facilities, distance_matrix, True, euclidean

def load_instance_Beasley(i:int, prefix: str = 'cap'):
    try:
        demand_facilities = pd.read_csv(f'{prefix}{i}.csv', index_col = [0,1])
        (f'{time.asctime()} - Instance {i} loaded from file')

    except:
        print(f'{time.asctime()} - File "{prefix}{i}.csv" not found.')    
    
    demand = demand_facilities.loc['Demand']
    demand = demand.rename({'capacity':'volume'}, axis = 1)
    facilities = demand_facilities.loc['Processing']
    
    try:
        demand['geometry'] = pd.read_csv(f'xy_sources_{prefix}{i}.csv', index_col = 0).apply(Point, axis = 1)
        facilities['geometry'] = pd.read_csv(f'xy_hubs_{prefix}{i}.csv', index_col = 0).apply(Point, axis = 1)
        demand = geopandas.GeoDataFrame(demand, geometry = 'geometry')
        facilities = geopandas.GeoDataFrame(facilities, geometry = 'geometry')
        facilities['cap_cost_ratio'] = facilities['capacity']/facilities['cost']
        coordinates_known = True
    except:
        coordinates_known = False
            
    try:
        distance_matrix = pd.read_csv(f'{prefix}{i}_dm.csv', index_col = 0).T
        distance_matrix.index = distance_matrix.index.astype(int)
        distance_matrix.columns = distance_matrix.columns.astype(int)
        euclidean = False
    except:
        print(f'{time.asctime()} - File "{prefix}{i}_dm.csv" not found. Using Euclidean Distances.')
        distance_matrix = get_euclidean_distances(facilities, demand)
        euclidean = True
        
    return demand, facilities, distance_matrix, coordinates_known, euclidean
    
def load_optimal_solution(i:int, prefix: str = '', n_facilities: int = 0):
    opt_flows = pd.read_csv(f'Optimal solution_F{prefix}{i:04d}.csv', index_col = [0,1]).squeeze("columns")
    opt_operation_vars = pd.read_csv(f'Optimal solution_O{prefix}{i:04d}.csv', index_col = 0, nrows = n_facilities).squeeze("columns")
    optimum = pd.read_csv(f'Optimal solution_O{prefix}{i:04d}.csv', index_col = 0, header = None, skiprows = n_facilities+1).squeeze("columns").iloc[0]
    
    return opt_flows, opt_operation_vars, optimum

def do_the_initialization(demand, facilities, distance_matrix, distance_matrix_T, alpha_start, bandwidth_start, gamma_start, number_of_sampled_dps = 2500, number_of_surrogates = 500, coordinates_known = True, euclidean = True, non_euclidean_inflation = 0.4):
    ### Record start time;
    start_time = time.time()

    ### Initialize the results dataframe;
    results_df = pd.DataFrame(index = ['problem','compressed_problem','compressed_prob_opt','operation_vars','clustering','comp_time','update_time'])

    ### If needed, sample points for allocation boundary computation;
    demand_sample = np.random.choice(demand.index, min(number_of_sampled_dps,len(demand)), replace = False)
                    
    ### For instances where coordinates are known, compute a distance matrix between all points and the reference demand points in the sample;
    if coordinates_known:
        try:
            distance_matrix_intra_demand = get_euclidean_distances(demand,demand.loc[demand_sample]) ### A Euclidean estimate is used for the distances between the demand points;
        except:
            coordinates_known = False ### We should proceed with the procedure to estimate distances under unknown DP coordinates;

    ### For instances where coordinates are not known, compute a distance matrix between all points and the reference demand points in the sample using a bounding procedure;
    if not coordinates_known:
        print(f'{time.asctime()} - Euclidean distances could not be computed, potentially due to absence of geographical data. Defaulting to bounding procedure.')
        distance_matrix_intra_demand = bounding_procedure_intra_demand_distances(demand_sample, distance_matrix_T) ### We apply a bounding procedure using the triangle inequality;
                    
    ### For instances where coordinates are known, we apply a naive bounding procedure to establish surrogate points at the geometric boundary of the instance. It may be beneficial to rotate the instance such that the bounding box surrounding is is the smallest;
    if coordinates_known:
        try:
            ### Try to add surrogate points to the distance matrices using the coordinates of the demand points and facilities;
            distance_matrix_intra_demand_aug, distance_matrix_T_aug, distance_matrix_T_aug_elev = add_surrogate_points(demand, facilities, distance_matrix_intra_demand, distance_matrix_T, number_of_surrogates, euclidean, non_euclidean_inflation= non_euclidean_inflation)
        except:
            coordinates_known = False ### We should proceed with the procedure to obtain surrogate points under unknown DP coordinates;
                    
    ### For instances where coordinates are not known, we apply a different procedure to estimate a geographical bounding box around the demand points;
    if not coordinates_known:
        print(f'{time.asctime()} - Euclidean surrogate points could not be added, potentially due to absence of geographical data. Defaulting to artificial box generation.')
        distance_matrix_intra_demand_aug, distance_matrix_T_aug, distance_matrix_T_aug_elev = add_surrogate_points_unknown_coordinates(distance_matrix_intra_demand, distance_matrix_T, number_of_surrogates)
                        
    ### Initialize the base model;
    prob, flows, operation_vars, flow_vars = standard_CFLP_formulation(demand,facilities, distance_matrix, False) 
                
    norm_factor = distance_matrix_T.max().max()

    alpha_start = pd.DataFrame(alpha_start, index = facilities.index, columns = demand.index)
    bandwidth_start = pd.DataFrame(max(bandwidth_start,0.025), index = facilities.index, columns = demand.index)
    gamma_start =  pd.Series(gamma_start, index = facilities.index)
                
    results_df.loc['update_time','init'] = time.time()-start_time ### Report timing statistics -> Total time spent during initialization
    print(f'Time spent during the initialization: {(time.time()-start_time):.2f} seconds')

    return  (alpha_start,
            bandwidth_start,
            gamma_start,
            norm_factor,
            prob, 
            flows, 
            operation_vars, 
            flow_vars,
            distance_matrix_intra_demand, 
            distance_matrix_intra_demand_aug, 
            distance_matrix_T_aug, 
            distance_matrix_T_aug_elev, 
            demand_sample,
            results_df,
            coordinates_known)
    
def compute_optimal_solution(problem_tuple, i, prefix = '', demand: geopandas.GeoDataFrame = None, facilities: geopandas.GeoDataFrame = None, epsilon = 0.001, valid_inequalities = True, plot = True):
    prob, flows, operation_vars, flow_vars = problem_tuple
    
    start_time = time.time()   

    if valid_inequalities:
        m = gppd.add_constrs(prob, operation_vars.sub(flow_vars,level = 0), GRB.GREATER_EQUAL, 0)

    prob.optimize()

    total_model_time = time.time()-start_time    

    open_facilities = facilities[operation_vars.apply(lambda x: (x-epsilon).getValue()) > 0]   
    optimum = prob.ObjVal
    opt_operation_vars = operation_vars.apply(lambda x: round((x+0).getValue()))
    print(f'{time.asctime()} - Optimal objective value : {optimum}')
    if plot:             
        demand.plot(markersize= demand['volume'], zorder = 30)
        open_facilities.plot(ax=plt.gca(),c='g', markersize=375, marker = 's')
        l = facilities.plot(ax = plt.gca(), marker = 's', column = facilities['capacity'].values, cmap = 'Wistia', markersize = 250)
        plt.colorbar(l.get_children()[2])
                            
        for of in open_facilities.index:
            xy_f = facilities.loc[of].geometry.xy
            current_flows = flows.loc[of]
            assigned_dps = current_flows[current_flows.apply(lambda x: (x-epsilon).getValue()) > 0].index
            xy_dps = demand.loc[assigned_dps].geometry.apply(lambda x: x.xy).values
                            
            for xy_dp in xy_dps:
                plt.plot([xy_f[0],xy_dp[0]],[xy_f[1],xy_dp[1]], c = '0.5', linewidth = 0.1, zorder = -10)
                                        
        plt.gcf().set_size_inches(10,10)
        plt.savefig(f'Solution instance {prefix}{i:04d}.png')

        plt.close()

    solution_table_fv = flows.apply(lambda x: (x+0).getValue())
    solution_table_ov = opt_operation_vars.copy()
    solution_table_ov.loc['Objective'] = optimum
    solution_table_fv.to_csv(f"Optimal solution_F{prefix}{i:04d}.csv")
    solution_table_ov.to_csv(f"Optimal solution_O{prefix}{i:04d}.csv")
    timing_table = pd.Series(index = ['Total','Runtime','WKU'])
    timing_table.loc[['Total','Runtime','WKU']] = [total_model_time, prob.Runtime, prob.Work]
    timing_table.to_csv(f"Optimal solution_T{prefix}{i:04d}.csv")
    
    if valid_inequalities:
        prob.remove(list(m))
    prob.reset(1)

    return optimum, opt_operation_vars

def obtain_benchmark_solution(demand, facilities, distance_matrix, i: int, run_prefix: str = '', n_clusters: int = 8, n_init_benchmark: int = 1, timeout: float = 1200):
    skm_result = solve_simple_kmeans(demand, facilities, distance_matrix, n_clusters = n_clusters, n_init = n_init_benchmark, timeout = timeout)
    benchmark_info = pd.Series(skm_result)

    benchmark_info.loc['problem'] = benchmark_info.loc['problem'].ObjVal
    benchmark_info.loc['compressed_problem'] = benchmark_info.loc['compressed_problem'].ObjVal
    benchmark_info.loc['operation_vars'] = benchmark_info.loc['operation_vars'].gppd.X.to_dict()  

    benchmark_info.loc[['problem','compressed_problem','operation_vars','clustering','time']].to_csv(f'{run_prefix}_Benchmark Instance {i:02d}.csv')
    simple_solution = (skm_result['problem'].ObjVal, skm_result['compressed_problem'].ObjVal)
    
    return simple_solution, benchmark_info

def create_and_save_animation(demand,
                              facilities,
                              clusterings,
                              operation_sols,
                              objective_values,
                              UB_vals,
                              iterations,
                              open_facilities,
                              i: int,
                              prefix = '',
                              run_prefix = '',
                              simple_solution = None,
                              coordinates_known = True,
                              n_init_benchmark = 0,
                              benchmark_multi_run = False,
                              compute_optimal_flag = False,
                              optimum = np.nan,
                              benchmark_min = np.nan,
                              benchmark_max = np.nan,
                              benchmark_min_c = np.nan,
                              benchmark_max_c = np.nan,
                              ):
    #### Delete cluster column if it exists
    try:
        del demand['cluster']
    except:
        pass
    if coordinates_known:
        fig1 = plt.figure()
        fig1.add_axes((0.05,0.05,0.425,0.9), xlim = (0,1), ylim = (-1,0))

        plt.box()

        fig1.add_axes((0.55,0.075,0.425,0.85), xlim = (0,iterations))

        fig1.axes[1].plot(*zip(*objective_values.items()))
        if compute_optimal_flag:
            fig1.axes[1].plot(*zip(*UB_vals.items()), c = 'turquoise')

        if n_init_benchmark > 0 or benchmark_multi_run:
            fig1.axes[1].axhline(simple_solution[1], c= 'g', linestyle = '--')
        if not np.isnan(optimum):
            fig1.axes[1].axhline(optimum, c= '0', linestyle = '--')
        if benchmark_multi_run:
            fig1.axes[1].axhspan(benchmark_min, benchmark_max, color = 'black', linestyle = '--', alpha = 0.15)

        if n_init_benchmark > 0 or benchmark_multi_run:
            fig1.axes[1].axhline(simple_solution[0], c= 'r', linestyle = ':')
        if benchmark_multi_run:
            fig1.axes[1].axhspan(benchmark_min_c, benchmark_max_c, color = 'black', linestyle = '--', alpha = 0.15)
        anim = animation.FuncAnimation(fig1, partial(animate, figure = fig1, demand = demand, facilities = facilities, clusterings = clusterings, operation_sols = operation_sols, objective_values = objective_values, iterations = iterations, open_facilities = open_facilities),
                                            frames=iterations, interval=1000, blit= True)
        try:
            del demand['cluster']
        except:
            pass
        
        writer = animation.PillowWriter(fps=1,
                                            metadata=dict(artist='Me'),
                                            bitrate=1800)

        try:
            anim.save(f'{run_prefix}_Algorithm_example_{prefix}{i}.gif', writer=writer)
        except:
            pass
    
    plt.close()
    
def plot_phi(ax, cax, demand, facility, distance_matrix, bwx, bw, alpha, bandwidth, gamma):
    transformed_space = (distance_matrix/distance_matrix.max().max()).apply(lambda x: phi(x, bwx, bw, bandwidth.loc[x.name],alpha.loc[x.name], gamma[x.name]), axis = 1).T
    demand['phi'] = transformed_space[facility.index[0]]
    demand.plot(ax = ax, column = 'phi', markersize = 3*0.8*demand['volume']/demand['volume'].mean(), legend = True, cax = cax, vmin=0, vmax=1)
    facility.plot(ax = ax, c = 'r', marker = 's', markersize = 25)
    
    ax.set_xticks([])
    ax.set_yticks([])

    del demand['phi']
    
    
####TODO: rewrite documentation
def animate_phi(demand: pd.DataFrame, facilities: pd.DataFrame, alpha: pd.DataFrame, distance_matrix: pd.DataFrame, bwx: np.ndarray, bw: np.ndarray, bandwidth: float = 0.3, gamma: float = 1.0, filename: str = 'phi_animation.gif'):
    """
    Creates an animation of the phi values for each facility in the facilities DataFrame.

    Parameters:
    demand (pd.DataFrame): DataFrame containing the demand points with a 'geometry' column.
    facilities (pd.DataFrame): DataFrame containing the facilities with a 'geometry' column.
    alpha (pd.DataFrame): DataFrame containing the alpha values for each facility.
    distance_matrix (pd.DataFrame): DataFrame containing the distances between demand points and facilities.
    bwx (np.ndarray): Array of beta values.
    bw (np.ndarray): Array of bandwidth values.
    bandwidth (float): The bandwidth parameter of the decay function.
    gamma (float): The gamma parameter of the decay function.
    """
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    plt.box()
    cax = divider.append_axes("right", size="5%", pad=0.1)

    def update(frame):
        ax.clear()
        cax.clear()
        facility = facilities.index[frame]
        plot_phi(ax, cax, demand, facilities.loc[[facility]], distance_matrix, bwx, bw, alpha, bandwidth, gamma)
        ax.set_title(f'Facility {facility}')

    anim = animation.FuncAnimation(fig, update, frames=len(facilities), interval=1000)

    # Save the animation as a GIF
    anim.save(filename, writer='pillow', fps=1)