# RF_Algorithm_Prediction
### Goal
Train random forests (one for each budget) that do multi-label classification. Given ELA features from one run and one budget, this forest should label all the A2 algorithms that achieve maximum precision for that switching point with 1. The purpose of this experiment is to visualize how the performance of such a selector changes throughout budgets.

### Model
Because for one budget of a run, there could be several algorithms that reach the best precision for that budget, I used a MultiOuputClassifier. In the evaluation however, I only considered the label that that Classifier assigns the highest probability and check if the algorithm corresponding to that label is optimal for the budget. I again that leave-instance-out cross-validation.

### Results
Inside results. Surprisingly, the accuracy is quite high for low budgets and then doesnt increace too significantly over budgets.