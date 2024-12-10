# CFLP-Aggregation

This project implements an iterative matheuristic algorithm for CFLPs

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- GeoPandas
- Shapely
- SciPy
- Gurobi
- gurobipy_pandas
- scikit-learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ruurdjb/CFLP-Aggregation
    ```

2. Install the required Python packages:
    ```sh
    pip install numpy pandas matplotlib geopandas shapely scipy gurobipy gurobipy_pandas scikit-learn
    ```

3. Make sure to keep the instance files on which you want to run the algorithm in the same directory as the files `iterative_algorithm.py` and `aggregation_helpers.py`
## Command Line Arguments

The following command line arguments should be passed to the script:

- `numrange`: The range of instance numbers to solve (e.g. for instances 1 up to and including 10, `1 10`).
- `fileset`: Different file formatting may be used for different instance sets. For the newly generated instances in our paper, use 'CF'.
- `alpha_start`: Initial value for alpha.
- `bandwidth_start`: Initial value for bandwidth.
- `gamma_start`: Initial value for gamma.
- `eta1`: Learning rate for alpha.
- `eta2`: Learning rate for bandwidth.
- `eta3`: Learning rate for gamma.
- `kappa_max`: Maximum number of best solutions to keep.
- `probability_forced_exploration`: Probability of forced exploration (after the warmup period).
- `max_forced_exploration`: Maximum number of new facilities seen to enforce during forced exploration.
- `clusters`: Number of clusters/ADPs.
- `iterations`: Maximum number of iterations.
- `warmup_duration`: Warmup period duration.
- `early_stopping_threshold`: Early stopping threshold: The algorithm should terminate when not improving the found solution for this number of iterations.
- `n_init_benchmark`: Number of initializations for the benchmark method. Recommended: 0 or 1. If a file summarizing multiple benchmark runs is present, or no benchmark run should be evaluated as a reference, pass 0. 
- `n_init`: Number of initializations for the clustering method. Recommended: 1
- `valid_inequalities`: Whether or not to add simple valid inequalities in the solution process.
- `compute_optimal_flag`: Flag indicating whether a pre-computed optimal solution should be used for reference or the optimal solution should be computed.
- `prefix`: Prefix for instance files.
- `run_prefix`: Prefix for output files corresponding to a particular run.
- `timeout`: Maximum time allowed for each instance, pass 0 for not imposing time limit. On Windows systems, it is highly recommended to pass 0 to avoid errors.
- `iteration_timeout`: Maximum time allowed for each instance, pass 0 for not imposing time limit. On Windows systems, it is highly recommended to pass 0 to avoid errors.
- `threads`: Number of threads the solver can use when solving an (MI)LP.
- `nodefiledir`: Directory where node files should be kept.
- `seed`: Random seed.

## Outputs
The program will output a log reporting the progress of the algorithm.

The program should write the following outputs to file for each instance:
- A CSV file containing the instance solution summary (`<run_prefix>Instance <instance_number>.csv`)
- An animation file containing a graphical depiction of the clusters/ADP, the objective of the reduced problem and final objective over the iterations (`<run_prefix>_Algorithm_example_<prefix><instance number>.gif`)
- Optionally, files containing the optimal solution and corresponding runtime (`Optimal solution_O<prefix><instance number>.csv`, `Optimal solution_F<prefix><instance number>.csv`, and `Optimal solution_T<prefix><instance number>.csv`)


## Example usage

```sh
python iterative_algorithm.py 0 9 'CF' 0 1 1 0.35 0.15 0.18 6 0.2 5 20 40 10 25 1 1 1 0 'P' 'C20' 9000 1200 10 './' 101224
```
