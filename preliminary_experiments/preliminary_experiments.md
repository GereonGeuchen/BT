# Preliminary Experiments
These are some first experiments to understand ELA feature behavior over budgets and the performance of warm-started algorithms over the budgets ("feature-view" and "performance-view" on switching optimality)

## Data Collection
The first step for the data collection is running the algorithms and logging their performance. This is done inside collect_data.py. I used the implementation from the per-run paper that is available in the zenodo repository.
I ran collect_A1 and collect_A2 inside collect_data.py with switching points at every 50 evaluations for an overall budget of 1000:

### A1 Data Collection
In collect_A1, the performance of the A1 algorithm CMA-ES is logged: The x- and the raw_y values for each function evaluation is logged, as well as the time-series internal state data (that I have not used so far). I then created csv files that only contain the necessary data to compute the ELA features over budgets: For each evaluation, there is a row containing: The fid, iid, rep, raw_y, true_y and x values, where $true\_y = raw\_y + function\_minimum$, as I noticed that raw_y only logs the precision of an evaluation but not the absolute objective values. I then ended up with 20 csv files (for each multiple of 50 until 1000) with each file containing all the evaluations of all the runs up to that budget. These files are so big that I cannot push them to github unfortunately. I then used the x and the true_y values to calculate ELA features (pflacco_computation.py) and stored them in A1_data_ela.

### A2 Data Collection
The A2 data contains the performances of all the warm-started A2 algorithms over the budgets. collect_A2 inside collect_data.py logs the performance of all runs in which we switch at different budgets to different algorithms. I collected that performance for budgets 50, 100, 150,...,1000 again.
I then created the file A2_precisions.csv, that for each switching point in each run contains the precision (reached within 1000 function evaluations)of all six A2 algorithms warm-started at that switching point. I specifically made sure that only values reached within 1000 evaluations are considered, as for some warm-started algorithms, one interation can account for hundreds of evaluations (e.g. in MLSL, multiple local searches are started in each iteration or BFGS using line-search). I also collected A2 data for all possible switching points (every eight evaluations because CMA has population size 8) and stored the precisions of the A2 algorithms over these switching points in A2_precision_all_switching_points.csv

## Feature clusters
In feature_clusters.py, I used PCS or MDS to create plots containing the clusters of ELA features over budgets, where either all features belonging to one function or all features belonging to on high_level_category are coloured the same in the plot. I excluded the .costs_runtime features as they do not capture landscape properties. Results in results/Clusters.

## Random forests 
Code in random_forests.py
Here, we ran an experiment if random forests are able to predict which function or high_level_category ELA features belong to over different budgets. We considered to cases:
1. A random forest that is trained on ELA data for which the samples are created using Latin Hypercube sampling. We first created the file ela_initial_sampling.csv in which we created initially sampled ELA features, again 20 times for the first five instances of the 24 fucntions each, where each initial sample contains 1000 points (to simulate the runs). The forests performance is then evaluated over the budget-specific ELA files to see how much initially sampled and run-based ELA features differ. Results in results/RF-results/RF_initial_fid.png. In the initial samples, the feature ela_meta.quad_simple.cond is sometimes inf, so we excluded it.

2. For each budget, trained and evaluated a random forest that predicts either fid or hlc using leave-instance-out cross validation, performance in results/RF-results/RF_budgets_fid.png. As always, ela_meta.quad_simple.adj_r2 is excluded for budget 50.

I repeated the experiments with normalized ELA features. In A1_data_ela_normalized, the features are normalized for each budget, in A1_data_ela_globally_normalized the normalization is applied across all budgets.

## ELA distributions
Code in ela_distributions.py, results in results/ela_distributions
For each feature and each function, plotted the distribution of that feature across all budgets (so over all five instances and twenty runs each). Also included the distribution of the feature when sampled initially.

## Precision analysis
In the performance-based evaluation of switching strategies, we aim to identify the most effective point in time to switch from the baseline algorithm (CMA-ES) to one of the six second-stage algorithms (A2 algorithms) purely based on the performance of the A2 algorithms.

The definition is as follows:

1. **Per-run Evaluation**  
   For each individual run (defined by a function ID, instance ID, and repetition), we consider a set of predefined switching points (e.g., every 50 evaluations up to a total budget of 1000).

2. **Best Precision at Each Switching Point**  
   At each switching point, we evaluate all six A2 algorithms, each warm-started from the current CMA-ES state. For each switching point, we select the **maximum precision** reached among the six A2 algorithms.  
   → This gives us one scalar value per switching point: the best final precision achievable **from that point onward** using any A2 algorithm.

3. **Optimal Switching Point**  
   Among all switching points in that run, we identify the one with the **highest precision** 
   → This switching point is considered the **optimal switching point** for that run, based on the actual performance of the warm-started algorithms.

We can thus identify optimal switching points per run. The obvious flaw of this optimality defintion is that it is purely performance-based and only considers the performance of the best A2 algorithm for each switching point. If we applied that definition but the selector is not able to choose the right A2 algorithm (which could really well happen for low budgets), we have gained nothing. However, it does give an understanding of the possible performance of A2 algorithms across budgets. Results are inside the folder results/precision_optimality.

In precision_optimality_50, you can see the results for experiments where I considered every 50 evaluations as a switching point. In precision_optimality_8 are the results where every 8 evaluations where considered as a switching point (so every possible, because eight is the population size).