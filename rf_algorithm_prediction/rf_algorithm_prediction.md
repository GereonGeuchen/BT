# RF_Algorithm_Prediction

## Goal

The objective of this experiment is to train **Random Forest models** (one per budget level) to perform **multi-label classification**. Each model predicts, given the ELA (Exploratory Landscape Analysis) features extracted from a single optimization run at a specific budget, which **A2 algorithms** would achieve the **highest precision** within 1000 evaluations if chosen at that switching point.

This setup allows us to analyze how well a model can predict good switching decisions based on ELA features and how this ability evolves across different budget levels. 

---

## Model Design

- For each budget, a **Random Forest classifier** is trained using a `MultiOutputClassifier`, because:
  - A single run at a given budget may have **multiple algorithms** that achieve the **best (maximum) precision**, requiring multi-label output.

- **Training setup**:
  - One forest per budget level.
  - Features: ELA features for that run and budget.
  - Targets: Binary labels indicating which A2 algorithms reach the best precision at that point.

- **Evaluation**:
  - For prediction, we only consider the **algorithm assigned the highest probability** by the model.
  - We then check whether that algorithm is among the **optimal choices** (i.e., those with maximum precision for that budget).
  - Evaluation is performed using **leave-one-instance-out cross-validation**, to test generalization across problem instances.
  - In accuracy_per_fid, you can find a plot for each budget in which the accuracy per fid is plotted. In the plots, you also find lines for each second algorithm that indicates the percentage of runs for which that second algorithm was best for the fid at that budget.



## Results
### Accuracy Trends Across Budgets
The plots show the accuracy over budgets (performance_over_budgets.pdf) and then the function-specific accuracy over budgets (in the folder accuracy_per_fid). In this plots, there is another line for each algorithm, where that line represents the percentage of switching points for each function, for which that algorithm was the optimal choice (so that is the algorithm the model should have predicted). So, if that line is at 100%, then for that function, that algorithm was the optimal choice for all runs. If it is at 50%, then for only 50% of runs, that algorithm was the optimal choice. A difficult function, also one where the model is quite bad, is function 17, in which there is no clear best algorithm for all runs. Here, the model is not able to distinguish between runs (see function-specific behavior).

The model achieves an accuracy of **64% at budget 50**, which is surprisingly high given the limited amount of information available at early stages. While the accuracy **increases for higher budgets**, it does so only moderately — reaching just above **80% at best**. This suggests that most of the predictive power is already captured by the ELA features at lower budgets.

In accuracy_per_fid, I also plotted how the accuracy changes for each fid over budgets, indicating different behaviour from function to function (which again motivates dynamic switching).

### Function-Specific Behavior

The plots indicate a strong link between the **distribution of optimal algorithms per function (fid)** and the **model's prediction accuracy**. In cases where a **single algorithm consistently performs best for a given function** — such as **BFGS for fids 9 to 12** — the model achieves **near-perfect accuracy (100%)**. This implies that the model’s predictions are largely **function-specific rather than run-specific**.

### Interpretation

If one algorithm is repeatedly optimal for a given function, the model learns to always predict that algorithm — leading to high accuracy. Therefore, part of the observed performance gain over budgets may not stem from better predictive modeling, but rather from **shifts in the distribution of best algorithms**.

For example, at **budget 950**, the accuracy curve closely resembles the distribution of **BFGS** being optimal across functions, indicating that the model increasingly follows trends in the data rather than run-level distinctions.

## Next steps
I would like to extend this approach and use the actual model type that we will use for our selector. I now that, in the per-run paper, regression models that predict the log-precision for each algorithm, were used. If we use them too, I would like to analyze their performance over budgets. If this approach performs similar to the model that I used so far, that would be quite interesting as it seems to perform really well, even for low budgets. A good analysis of the performance over budgets would allow us to get closer to the optimality definition that we want to use. If the performance does not change significantly over budgets, that definition should favor the "performance"-view (so at which switching point are the A2 algorithms best) more than the "feature"-view (at which switching point is the model that chooses the A2 algorithm best). In terms of the storyline of the thesis, I think it would also be really nice to see how these to "views" develope over budgets side by side.
