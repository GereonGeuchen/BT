import numpy as np
import LassoBench
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))
from bfgs import BFGS # type: ignore
from pso import PSO # type: ignore
from mlsl import MLSL # type: ignore
from de import DE # type: ignore

class LassoBenchWrapper:
    class _State:
        def __init__(self):
            self.evaluations = 0

    class _Bounds:
        def __init__(self, dim):
            self.lb = np.full(dim, -1.0, dtype=float)
            self.ub = np.full(dim, 1.0, dtype=float)

    def __init__(self, bench):
        """
        bench: an instance of RealBenchmark (from LassoBench)
        """
        self.bench = bench
        # self.dim = getattr(bench, "n_features", None)
        self.dim = bench.n_features
        if self.dim is None:
            # Try to infer from alpha range; if not possible, keep None
            self.dim = bench.X_train.shape[1]

        # Bounds in the search space expected by LassoBench.evaluate()
        self.bounds = self._Bounds(self.dim)

        self.state = self._State()
        self._logger = None

    def reset(self):
        """Reset internal evaluation counter."""
        self.state.evaluations = 0

    # def attach_logger(self, logger):
    #     """Attach a logger (IOH or custom)."""
    #     self._logger = logger
    #     # If the logger provides a reset/init, call it.
    #     for m in ("reset", "initialize"):
    #         if hasattr(self._logger, m) and callable(getattr(self._logger, m)):
    #             try:
    #                 getattr(self._logger, m)()
    #             except Exception:
    #                 pass

    # def detach_logger(self):
    #     """Detach any logger."""
    #     self._logger = None

    # def _log(self, x, y):
    #     """Best-effort logger compatibility."""
    #     if self._logger is None:
    #         return
    #     # Try common patterns (adjust this to your IOH logger if needed)
    #     try:
    #         if hasattr(self._logger, "add") and callable(self._logger.add):
    #             # e.g., logger.add(evals, y, x_list)
    #             self._logger.add(self.state.evaluations, float(y), list(map(float, x)))
    #         elif hasattr(self._logger, "log") and callable(self._logger.log):
    #             # e.g., logger.log(evals, y, x_array)
    #             self._logger.log(self.state.evaluations, float(y), np.asarray(x, dtype=float))
    #         elif callable(self._logger):
    #             # e.g., logger(evals, y, x)
    #             self._logger(self.state.evaluations, float(y), np.asarray(x, dtype=float))
    #     except Exception:
    #         # Don't let logging failures break evaluations
    #         pass

    def __call__(self, x):
        """
        Evaluate one configuration x (array-like of shape [dim]).
        Returns a float (CV loss).
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected x.shape == ({self.dim},), got {x.shape}")

        # (Optional) clip to bounds to avoid ValueError from LassoBench
        # if np.any(x < self.lb) or np.any(x > self.ub):
        #     raise ValueError("x is outside [-1, 1]^d required by LassoBench.")

        y = float(self.bench.evaluate(x))  # RealBenchmark.evaluate returns CV loss (scalar)

        # increase evaluation count, log
        self.state.evaluations += 1
        print(self.state.evaluations)
        # self._log(x, y)
        print(y)
        return y
    

    def __getattr__(self, attr):
        return getattr(self.bench, attr)
    

# real_bench = LassoBench.RealBenchmark(pick_data='Diabetes')
# d = real_bench.n_features
# # random_config = np.random.uniform(low=-1.0, high=1.0, size=(d,))
# # loss = real_bench.evaluate(random_config)
# # print(real_bench.n_features)  # Should print the number of features
# # print(loss)
# problem = LassoBenchWrapper(real_bench)
# algorithm = DE(problem, verbose = False, seed = 42)

# def stopping_criteria():
#     return problem.state.evaluations >= 1000

# algorithm.set_stopping_criteria(stopping_criteria)

# algorithm.run()
# # If stored:
# print("Known optimum value:", getattr(problem.bench, "fopt", None))
# print("Known optimum location:", getattr(problem.bench, "xopt", None))

import numpy as np
import LassoBench
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

# --- your existing setup ---
real_bench = LassoBench.RealBenchmark(pick_data='Breast_cancer')
problem = LassoBenchWrapper(real_bench)

# ---------- tiny grid-search (no Sobol, just loops) ----------
def simple_grid_search(problem,
                       budget=1000,
                       subset_dims=32,  # how many coordinates to sweep
                       grid_points=5,   # how many values per sweep (linspace on [-1,1])
                       passes=1,        # do multiple passes to refine
                       start=None):
    d = problem.dim
    # choose a subset of coordinates to sweep (first N; change to random if you prefer)
    idx = np.arange(min(subset_dims, d))

    # start at zero (center) unless user provided a start point
    x = np.zeros(d, dtype=float) if start is None else np.asarray(start, dtype=float)
    x = np.clip(x, -1.0, 1.0)

    # evaluate start
    best_y = problem(x)
    best_x = x.copy()

    # coarse grid values (always within [-1,1])
    vals = np.linspace(-1.0, 1.0, grid_points)

    for _ in range(passes):
        for j in idx:
            # coordinate sweep on dimension j
            x_local = best_x.copy()
            local_best_y = best_y
            local_best_v = x_local[j]
            for v in vals:
                if problem.state.evaluations >= budget:
                    return best_x, best_y
                x_local[j] = v
                y = problem(x_local)
                if y < local_best_y:
                    local_best_y = y
                    local_best_v = v
            # lock in the best value found for coordinate j
            best_x[j] = local_best_v
            best_y = local_best_y

        # optional tiny refinement around the current best (still only loops)
        if grid_points >= 5:
            vals = np.linspace(-0.5, 0.5, grid_points)  # shrink the step
            vals = np.clip(vals + 0.0, -1.0, 1.0)       # keep in bounds

        if problem.state.evaluations >= budget:
            break

    return best_x, best_y

# run it
best_x, best_y = simple_grid_search(problem,
                                    budget=100000,
                                    subset_dims=128,  # tweak to 64/128 if you want
                                    grid_points=5,
                                    passes=20)

print("\nGrid-search done.")
print("Evaluations used:", problem.state.evaluations)
print("Best CV loss found:", best_y)
print("Best x (first 10 dims):", np.array2string(best_x[:10], precision=3))
print("Known optimum value:", getattr(problem.bench, "fopt", None))
print("Known optimum location:", getattr(problem.bench, "xopt", None))
