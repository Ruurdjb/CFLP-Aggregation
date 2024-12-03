import numpy as np
import pandas as pd
import time
import argparse
import gurobipy as gp

from aggregation_helpers import *

def main():
    #### Initialize reference arrays for decay function beta computation
    granularity_bandwidth_comp = 600
    bwx = np.concatenate([np.linspace(3.5,10,int(granularity_bandwidth_comp/4)), np.linspace(10,25,int(granularity_bandwidth_comp/4)), np.linspace(25,300,int(granularity_bandwidth_comp/4)), np.linspace(300,2000,int(granularity_bandwidth_comp/4))])
    bw = 2*np.log(bwx+np.sqrt(bwx*(bwx-2)) - 1)/bwx
       
    for i in range(n_start,n_stop+1):
        try:
            with time_limit(timeout):
                print(f'{time.asctime()} --- START INSTANCE {i} ---')
                iteration = 0 
                instance_loaded = False
                try:
                    demand, facilities, distance_matrix, coordinates_known, euclidean = load_instance(fileset, i, prefix)
                    instance_loaded = True
                except:
                    print(f'{time.asctime()} - Instance {i} could not be loaded. The instance will be skipped.')
                
                if instance_loaded:     
                
                    if coordinates_known:    
                        plot_problem_instance(demand, facilities,f'Instance_{prefix}{i:04d}.png')
                    
                    print(f'{time.asctime()} --- STARTING INITIALIZATION ---')
                    
                    distance_matrix_T = distance_matrix.T
                    number_of_surrogates = 500 ##This is best divisible by 4.
                    number_of_sampled_dps = 2500 ### This is an arbitrary number, should not be too small (otherwise we get a very rough allocation boundary) but also not too large (this would be too heavy computationally);
                        
                    (alpha, 
                    bandwidth, 
                    gamma,    
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
                    coordinates_known) = do_the_initialization(demand, facilities, distance_matrix, distance_matrix_T, alpha_start, bandwidth_start, gamma_start, number_of_sampled_dps, number_of_surrogates, coordinates_known, euclidean, non_euclidean_inflation)

                    print(f'{time.asctime()} --- Reference point computation complete. INITIALIZATION FINISHED ---')

                    epsilon = 0.001   

                    try:
                        opt_flows, opt_operation_vars, optimum = load_optimal_solution(i, prefix, len(facilities))
                        open_facilities = facilities[opt_operation_vars>epsilon]
                        print(f'{time.asctime()} - Optimal solution to instance {i} loaded from file')
                    except:
                        if compute_optimal_flag:
                            print(f'{time.asctime()} - Files "Optimal solution_F{prefix}{i:04d}.csv" and/or "Optimal solution_O{prefix}{i:04d}.csv" not found or could not be loaded. The optimum for this instance will now be computed.')
                            optimum, opt_operation_vars = compute_optimal_solution((prob, flows, operation_vars, flow_vars), i, prefix, demand, facilities, valid_inequalities = valid_inequalities, plot = coordinates_known)
                            open_facilities = facilities[opt_operation_vars>epsilon]
                        else:
                            print(f'{time.asctime()} - Files "Optimal solution_F{prefix}{i:04d}.csv" and/or "Optimal solution_O{prefix}{i:04d}.csv" not found or could not be loaded. The optimum for this instance will not be computed.')
                            open_facilities = None
                            opt_operation_vars = None
                            optimum = np.nan

                    benchmark_multi_run = False
                    if n_init_benchmark > 0:
                        simple_solution, benchmark_info = obtain_benchmark_solution(demand, facilities, distance_matrix, i, run_prefix, clusters, n_init_benchmark, iteration_timeout)
                        print(f'{time.asctime()} - Objective value reference aggregation: {simple_solution[0]}')
                    else:
                        try:
                            benchmark_summary = pd.read_csv(f"{run_prefix}_Benchmark Instance {prefix}{i:04d}.csv", index_col = 0)
                            simple_solution = (benchmark_summary.mean().values)
                            benchmark_min, benchmark_min_c = (benchmark_summary.min().values)
                            benchmark_max, benchmark_max_c = (benchmark_summary.max().values)
                            benchmark_multi_run = True
                            
                        except:
                            print(f'{time.asctime()} - File "_Benchmark Instance {prefix}{i:04d}.csv" could not be loaded. No benchmark info will be used.')
                            simple_solution = None,
                            benchmark_min, benchmark_min_c = (np.nan, np.nan)
                            benchmark_max, benchmark_max_c = (np.nan, np.nan)

                    objective_values = {}
                    clusterings = {}
                    operation_sols = {}
                    UB_vals = {}
                    
                    ### Make sure dynamic alpha is initialized
                    dynamic_alpha = alpha 
                    
                    solutions_list = pd.DataFrame({s : pd.Series(dtype = float) for s in facilities.index.append(pd.Index(['ObjVal']))} )
                    
                    for iteration in range(1, iterations+1): 
                        print(f'{time.asctime()} --- STARTING ITERATION {iteration} ---')

                        result = solve_dm_transformed_kmeans(demand,
                                                            facilities,
                                                            bwx,
                                                            bw, 
                                                            n_clusters = clusters, 
                                                            alpha = alpha, 
                                                            bandwidth = bandwidth, 
                                                            gamma = gamma, 
                                                            distance_matrix = distance_matrix, 
                                                            opt_operation_vars = opt_operation_vars, 
                                                            n_init = n_init, 
                                                            solutions_seen = solutions_list, 
                                                            forced_exploration = np.random.rand() < probability_forced_exploration if iteration > warmup_duration else 1, 
                                                            max_forced_exploration = max_forced_exploration,
                                                            problem_tuple = (prob,flows,operation_vars),
                                                            iteration_timeout = iteration_timeout)
                        
                        print(f'{time.asctime()} - Clustering terminated')
                        
                        ### Store results in a series
                        results_iteration = pd.Series(result)

                        ### Store the objective values of the problems in the results dataframe
                        results_iteration.loc['problem'] = results_iteration.loc['problem'].ObjVal
                        results_iteration.loc['compressed_problem'] = results_iteration.loc['compressed_problem'].ObjVal
                        
                        ### If the optimal solution is known, store its objective value in the compressed problem
                        if compute_optimal_flag:
                            results_iteration.loc['compressed_prob_opt'] = results_iteration.loc['compressed_prob_opt'].ObjVal 
                        else:
                            results_iteration.loc['compressed_prob_opt'] = np.nan
                        
                        ### Store the operation variable values in the results dataframe
                        results_iteration.loc['operation_vars'] = results_iteration.loc['operation_vars'].gppd.X.to_dict()  

                        ### Start the update timer
                        start_time = time.time()

                        alpha = result['alpha'] ##To make sure alpha is a dataframe
                        
                        compressed_flows = result['compressed_flows'].gppd.get_value().unstack()
                        compressed_dm = result['compressed_dm']

                        flow_vals = result['flows'].gppd.get_value().unstack()
                        operation_vars_vals = result['operation_vars'].gppd.X 

                        solutions_list.loc[iteration, operation_vars.index] = operation_vars_vals
                        solutions_list.loc[iteration, 'ObjVal'] = result['problem'].ObjVal
                        
                        if solutions_list.ObjVal.idxmin() <= iteration - early_stopping_threshold:
                            print(f'{time.asctime()} --- ALGORITHM TERMINATED DUE TO EARLY STOPPING CRITERION ---')
                            break 
                    
                        best_solutions_list = solutions_list.sort_values('ObjVal').iloc[:kappa_max,:]
                        
                        if phi_animation_flag:
                            animate_phi(demand, facilities, alpha, distance_matrix, bwx, bw, bandwidth, gamma, f'phi_iteration{iteration}.gif')

                        ###Update alpha
                        alpha_reference = pd.concat([obtain_alpha_reference(f, flow_vals.idxmax(axis = 0), distance_matrix_T_aug, distance_matrix_intra_demand_aug, number_of_surrogates = number_of_surrogates, sample_demand = demand_sample, epsilon_local_inward = 0.8/(1+non_euclidean_inflation*euclidean), epsilon_local_outward = 0.8/(1+non_euclidean_inflation*euclidean)).to_frame(f) for f in facilities[operation_vars_vals ==1 ].index], axis = 1)

                        dynamic_alpha = update_alpha(dynamic_alpha, iteration, alpha_reference, distance_matrix_T_aug_elev, norm_factor, eta = eta1)
                        alpha = dynamic_alpha if iteration >= warmup_duration else alpha_start  

                        bandwidth = update_bandwidth(bandwidth, iteration, eta = eta2) 
                        
                        gamma = update_gamma(gamma, iteration, best_solutions_list, eta = eta3, warmup_duration = warmup_duration)


                        print(f'{time.asctime()} - Updated alpha computed')

                        ### Store objectives, clusterings and operation variable solutions
                        objective_values[iteration] = (result['problem'].ObjVal, result['compressed_problem'].ObjVal)
                        clusterings[iteration] = result['clustering']
                        operation_sols[iteration] = facilities[result['operation_vars'].apply(lambda x: (x-epsilon).getValue()) > 0]

                        if compute_optimal_flag:
                            UB_vals[iteration] = result['compressed_prob_opt'].ObjVal

                        results_iteration.loc['update_time'] = time.time()-start_time
                        results_df[iteration] = results_iteration.loc[['problem','compressed_problem','compressed_prob_opt','operation_vars','clustering','comp_time','update_time']]

                        reference_solution_string = '' + f'| Reference {simple_solution[0]}' if n_init_benchmark > 0 else '' + f'| Reference (mean) {simple_solution[0]}' if benchmark_multi_run else '' 
                        
                        result['problem'].reset(1)

                        print(f'{time.asctime()} - ITERATION {iteration} Objective value: {objective_values[iteration][0]} {reference_solution_string} - Optimal {optimum}')
                        
                        print(f'{time.asctime()} --- ITERATION {iteration} TERMINATED ---')
        except TimeoutException as e:
            print(f"Instance {i} Timed out during ITERATION {iteration}!")
        
        if instance_loaded:
            results_df.to_csv(f'{run_prefix}Instance {i:02d}.csv')
            
            create_and_save_animation(demand, facilities, clusterings, operation_sols, objective_values, UB_vals, iterations, open_facilities, i, prefix, run_prefix, simple_solution, coordinates_known, n_init_benchmark, benchmark_multi_run, compute_optimal_flag, optimum, benchmark_min, benchmark_max, benchmark_min_c, benchmark_max_c)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'file name parser')
    parser.add_argument('numrange', metavar = 'N', type = int, nargs = 2)
    parser.add_argument('fileset', metavar = 'F', type = str, nargs = 1)
    parser.add_argument('alpha_start', metavar = 'A', type = float, nargs = 1)
    parser.add_argument('bandwidth_start', metavar = 'B', type = float, nargs = 1)
    parser.add_argument('gamma_start', metavar = 'G', type = float, nargs = 1)
    parser.add_argument('eta1', metavar = 'E1', type = float, nargs = 1)
    parser.add_argument('eta2', metavar = 'E2', type = float, nargs = 1)
    parser.add_argument('eta3', metavar = 'E3', type = float, nargs = 1)
    parser.add_argument('kappa_max', metavar = 'K', type = int, nargs = 1)
    parser.add_argument('probability_forced_exploration', metavar = 'PFX', type = float, nargs = 1)
    parser.add_argument('max_forced_exploration', metavar = 'MFX', type = int, nargs = 1)
    parser.add_argument('clusters', metavar = 'C', type = int, nargs = 1)
    parser.add_argument('iterations', metavar = 'T', type = int, nargs = 1)
    parser.add_argument('warmup_duration', metavar = 'WD', type = int, nargs = 1)
    parser.add_argument('early_stopping_threshold', metavar = 'EST', type = int, nargs = 1)
    parser.add_argument('n_init_benchmark', metavar = 'BX', type = int, nargs = 1) ##Number of inits in  clustering method (Benchmark)
    parser.add_argument('n_init', metavar = 'I', type = int, nargs = 1) ##Number of inits in  clustering method
    parser.add_argument('valid_inequalities', metavar = 'VI', type = int, nargs = 1) ##Number of inits in  clustering method
    parser.add_argument('compute_optimal_flag', metavar = 'O', type = int, nargs = 1)
    parser.add_argument('prefix', metavar = 'P', type = str, nargs = 1)
    parser.add_argument('run_prefix', metavar = 'RP', type = str, nargs = 1)
    parser.add_argument('timeout', metavar = 'TO', type = int, nargs = 1)
    parser.add_argument('iteration_timeout', metavar = 'ITO', type = int, nargs = 1)
    parser.add_argument('threads', metavar = 'TD', type = int, nargs = 1)
    parser.add_argument('nodefiledir', metavar = 'D', type = str, nargs =1)
    parser.add_argument('seed', metavar = 'S', type = int, nargs = 1)

    parsed_args = parser.parse_args()
    n_start, n_stop = parsed_args.numrange
    fileset, = parsed_args.fileset
    alpha_start, = parsed_args.alpha_start
    bandwidth_start, = parsed_args.bandwidth_start
    gamma_start, = parsed_args.gamma_start
    eta1, = parsed_args.eta1
    eta2, = parsed_args.eta2
    eta3, = parsed_args.eta3
    kappa_max, = parsed_args.kappa_max
    probability_forced_exploration, = parsed_args.probability_forced_exploration
    max_forced_exploration, = parsed_args.max_forced_exploration
    clusters, = parsed_args.clusters
    iterations, = parsed_args.iterations
    warmup_duration, = parsed_args.warmup_duration
    early_stopping_threshold, = parsed_args.early_stopping_threshold
    n_init_benchmark, = parsed_args.n_init_benchmark
    n_init, = parsed_args.n_init
    valid_inequalities, = parsed_args.valid_inequalities
    compute_optimal_flag, = parsed_args.compute_optimal_flag
    prefix, = parsed_args.prefix
    run_prefix, = parsed_args.run_prefix
    timeout, = parsed_args.timeout
    iteration_timeout, = parsed_args.iteration_timeout
    threads, = parsed_args.threads
    nfd, = parsed_args.nodefiledir
    seed, = parsed_args.seed
    phi_animation_flag = False

    np.random.seed(seed)
    
    gp.setParam('NodeFileStart', 0.000005)
    gp.setParam('NodeFileDir', f'{nfd}/grbnodes_ax_{prefix}{n_start}_{n_stop}')
    gp.setParam('OutputFlag', 0)
    gp.setParam('IntFeasTol',1/10**7)
    gp.setParam('Threads', threads)
    
    non_euclidean_inflation = 0.4
    main()