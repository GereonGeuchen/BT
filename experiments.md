# File Describing All the Experiments and Computations

## 1. Pflacco ELA Computation

Because we do not use ELA features as originally intended (e.g., often computing them with fewer samples than recommended, and using samples that are not well-distributed—especially when CMA-ES has converged, which is typically the case at high budgets), I had to consider a few adjustments to ensure that we can still use ELA features meaningfully.

### Key Considerations

1. **Budget = 50 (Too Few Samples)**  
   For this budget, the sample size is insufficient to compute certain features reliably. Therefore:
   - In the `ela_level` feature group, features using **quantile 0.1** are **not computed**.
   - The feature `ela_meta.quad_w_interact.adj_r2` returns **NaN values** due to a **division by zero**.  
     → I specifically **excluded this feature** from later uses like switching prediction.

2. **`ela_level` (Too Few Samples in Certain Quantiles)**  
   In the `ela_level` feature group, the objective values are split into quantiles. If CMA-ES has converged, there are many best objective values, which can result in **empty quantiles**.  
   In the code:
   ```python
   y_quant = np.quantile(y, prob)
   y_class = (y < y_quant)
   ```
   I added:
   ```python
   if y_class.sum() < ela_level_resample_iterations:
     y_class = (y <= y_quant)
   ```

3. **`dispersion` (Distances in Certain Quantiles Become Zero if CMA-ES Has Converged)**  
   In the `dispersion` feature set, the average pairwise distances between the best-performing samples (e.g., top 2%, 5%, etc.) are calculated. These distances are used to estimate how spread out the best solutions are compared to the overall sample.

   **Problem:**  
   If CMA-ES has converged for a long time, many of the best samples are identical or nearly identical. This causes:
   - All pairwise distances among them to be **zero**
   - The computed array of non-zero distances to be **empty**
   - Statistical computations like `mean(nonzero_dists)` or `median(nonzero_dists)` to fail

   **Fix:**  
   To avoid crashes or misleading feature values, I added a fallback in the code:
   ```python
   if nonzero_dists.size == 0:
       means.append(0.0)
       medians.append(0.0)

## 2. Preliminary Experiments
See preliminary_experiments/preliminary_experiments.md

## 3. rf_switching_prediction
The experiments in rf_switching_prediction are really just a small first step to see if we can use ELA features to predict "performance-based" switching points, see rf_switching_prediction/rf_switching_prediction.md