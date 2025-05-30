# RF_switching_prediction
In this folder, I did some experiments to see if random forests can be used to detect switching points. However, the optimality definition that is used here is purely performance-based and the random forests are not optimized (only the decision threshold), so this is really just a small first experiment. For each run, an optimal switching point is defined as follows: I consider every 50 evaluations as a switching point. For each of these switching points, I consider the precision of the best out of the six warm-started algorithm at that switching point. I then define the optimal switching point for each run as the switching point for which that precision is the lowest across all switching points. So this definition is purely performance-based and only considers the performance of the best algorithm for each switching point. The obvious flaw is that, if we would be able to detect these switching points perfectly, then we would still need the selector to actually choose the right algorithm.

## Content of this folder
### data
    -ELA_over_budgets: Contains one file for each budget, with one row for each recorded run (so 2400 rows, $24*5*20$ for 24 functions, 5 instances and 20 runs each).

    -ELA_over_budgets_with_optimality: Same content as ELA_over_budgets, but now their is a row "is_minimal_switch" where is_minimal_switch=true if the run has an optimal switching point at that budget (each row corresponds to a run)

    -ELA_over_budgets_with_precs: Same content, but with an additional row "run_precision". For each switching point in each run, the run_precision is defined as the difference between the precision of the best A2 algorithm for that switching point and the precision of the best A2 algorithm of the optimal switching point for that run. I use this file to turn the binary prediction (optimal switching point yes/no) to a regression problem.

### results
Contains the results of the experiments. In results/binary_switching_decisions are the results for the binary classification problem (optimal switching point yes/no), inside precision_regression are the results for the regression problem (how close is each switching point to the optimal switching point for a run).

### scripts
Contains the scripts that were used to run the experiments.

## Results: binary_switching_decision
To get the heatmaps and performances for the random forests for the binary switching decision, I did leave-instance out cross validation. So, a random forest is trained on four of the five instance across all functions and then we look at the predictions for the remaining instance. We do this five times to get predictions for all instances. One random forest is trained and evaluated for one budget each. In rf_heatmaps, you can see, for each function and instance, one heatmap with the predicted switching point probability for each budget and run on that instance. In binary_switch_heatmaps_rf, you can see, for each budget and each run, if a switching point was predicted for that budget. I used the decision thresholds in optimal_thresholds.txt that were chosen to maximize F1_score for each budget. The plots on the left are the predicted switching points, in the middle are the true switching points, on the right is the predicted switching probability.

## Results: precision regression
Instead of predicting if we have reached an optimal switching point or not, I turned the problem into a regression problem. For each possible switching point, I want to predict how close that point is to the optimal switching point. So, I predict the difference of the precision of the best second algorithm for that switching point and the precision of the best algorithm of the optimal switching point. You find the results in results/precision_regression. I recommend looking into the folder rf_run_precision_heatmaps. There you can find the predicted precision for each run and budget on each instance, with a heatmap with the true run precision on the right (so left predicted, right truth).

## switching_point_ahead:
Here I wanted to see what happens where instaed of predicting if a certain budget is an optimal switching point, I predict if an optimal switching is ahead for a certain budget (so e.g. if we are at budget 100, switching_point_ahead would be true if there was an optimal switching point at budget 500 for that run). I would then switch as soon as switchin_point_ahead is false. This did not lead to anything so you can just ignore it as of right now.