# Selector

Let's say we have $n$ switching points overall. Then, for the $i$-th switching point, we train $n-(i+1)$ regression models that predict the performance of the upcoming switching point. If the performance of the current is best, we switch. We additionally train 6 regression models per switching point that predict the performance of each algorithm and then use these 6 regression models to choose the algorithm after switching.

## Steps
1. Train the predictors of upcoming switching points for each switching point
2. Train the algorithm performance predictors for each switching point
3. Record the performance of that approach
4. Compare it to the performance of the VBS (VBS as defined by Hadar)