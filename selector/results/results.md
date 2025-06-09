# Current results of the selector

### vbs_precision_ratios.png

For every run, the performance of the vbs is defined as the minimal precision reached by one of the algorithms across all possible switching budgets. The precision ratio for the selector is then defined as the ratio between the vbs precision and the precision reached by the selector. For each static budget, the ratio is defined as the precision by the algorithm chosen by the selector of that budget and the vbs precision. We then average these values over all test runs (24 functions, instances 6 and 7, 20 run each) and get the final values displayed in the table.

### vbs_selector_ratios.png

For each run, the vbs is defined differently. We now only consider the performance of the chosen algorithms across budgets and the vbs performance is then the performance of the best chosen algorithm across budgets. The values are then computed in the same way as above.

### selector_budget_counts.png

Shows how often which budget was chosen by the selector.
