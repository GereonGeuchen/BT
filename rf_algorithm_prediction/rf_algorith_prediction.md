# RF_Algorithm_Prediction

## Goal

The objective of this experiment is to train **Random Forest models** (one per budget level) to perform **multi-label classification**. Each model predicts, given the ELA (Exploratory Landscape Analysis) features extracted from a single optimization run at a specific budget, which **A2 algorithms** would achieve the **highest precision** within 1000 evaluations if chosen at that switching point.

This setup allows us to analyze how well a model can predict good switching decisions based solely on ELA features — and how this ability evolves across different budget levels.

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

The model achieves an accuracy of **64% at budget 50**, which is surprisingly high given the limited amount of information available at early stages. While the accuracy **increases for higher budgets**, it does so only moderately — reaching just above **80% at best**. This suggests that most of the predictive power is already captured by the ELA features at lower budgets.

### Function-Specific Behavior

The plots indicate a strong link between the **distribution of optimal algorithms per function (fid)** and the **model's prediction accuracy**. In cases where a **single algorithm consistently performs best for a given function** — such as **BFGS for fids 9 to 12** — the model achieves **near-perfect accuracy (100%)**. This implies that the model’s predictions are largely **function-specific rather than run-specific**.

### Interpretation

If one algorithm is repeatedly optimal for a given function, the model learns to always predict that algorithm — leading to high accuracy. Therefore, part of the observed performance gain over budgets may not stem from better predictive modeling, but rather from **shifts in the distribution of best algorithms**.

For example, at **budget 950**, the accuracy curve closely resembles the distribution of **BFGS** being optimal across functions, indicating that the model increasingly follows trends in the data rather than run-level distinctions.
