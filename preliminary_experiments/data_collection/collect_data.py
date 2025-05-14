import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

import shutil
import argparse
from dataclasses import dataclass, fields
import pandas as pd

import ioh
from ioh import ProblemClass
from modcma import ModularCMAES, Parameters
import numpy as np

from bfgs import BFGS # type: ignore
from pso import PSO # type: ignore
from mlsl import MLSL # type: ignore
from de import DE # type: ignore
import warnings
from itertools import product
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool

def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(cpu_count(), len(arguments)))
#     local_func = partial(func_star, func=runFunction)
    results = p.map(runFunction, arguments)
    p.close()
    return results

@dataclass
class TrackedParameters:
    # Static meta info
    rep: int = -1
    iid: int = -1

    # Time series features
    sigma: float = 0
    t: int = 0
    d_norm: float = 0
    d_mean: float = 0 
    ps_norm: float = 0
    ps_mean: float = 0
    pc_norm: float = 0
    pc_mean: float = 0
    
    # Anja parameters:
    ps_ratio: float = 0
    ps_squared: float = 0
    loglikelihood: float = 0
    
    # check if this should only be one parameter
    mhl_norm: float = 0
    mhl_sum: float = 0
    
    def update(self, parameters: Parameters):
        self.sigma = parameters.sigma
        self.t = parameters.t
        for attr in ('D', 'ps', 'pc'):
            setattr(self, f'{attr}_norm'.lower(), np.linalg.norm(getattr(parameters, attr)))
            setattr(self, f'{attr}_mean'.lower(), np.mean(getattr(parameters, attr)))

        self.ps_squared = np.sum(parameters.ps ** 2)
        self.ps_ratio = np.sqrt(self.ps_squared) / parameters.chiN

        sigma2 = self.sigma ** 2
        
        if hasattr(parameters.population, "x"):
            delta = parameters.population.x.T - parameters.m.T
            self.loglikelihood = -.5 * (parameters.lambda_ * (
                parameters.d * np.log(2 * np.pi * sigma2) + np.log(np.prod(parameters.D) ** 2)) 
                + np.diag(delta.dot(parameters.inv_root_C / sigma2).dot(delta.T)).sum()                
            )
        else:
            delta = np.zeros((5, parameters.d))
            self.loglikelihood = 0        
        
        mhl = np.sqrt(
            np.power(np.dot(parameters.B.T, delta.T) / parameters.D, 2).sum(axis=0)
        ) / self.sigma
        self.mhl_norm = np.linalg.norm(mhl)
        self.mhl_sum = mhl.sum()

            
class TrackedCMAES(ModularCMAES):
    def __init__(self, tracked_parameters = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_parameters = tracked_parameters
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)
        
    def step(self):
        # self.mutate()
        # self.select()
        # if self.tracked_parameters is not None:
        #     self.tracked_parameters.update(self.parameters)
        # self.recombine()
        # self.parameters.adapt()
        # self.tracked_parameters.t = self.parameters.t
        # return not any(self.break_conditions)
        res = super().step()
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)
        return res 
            
class From_CMA_To_CMA():
    def __init__(self, budget_factor, dim, A2, total_budget_factor=200):
        self.budget_factor = budget_factor
        self.dim = dim
        self.A2 = A2
        self.total_budget = total_budget_factor*self.dim
        
    def __call__(self, problem, A2, hparams = {}):
        if A2 == "Same":
            budget = self.total_budget
        else:
            budget = self.dim * self.budget_factor
            
        cma = TrackedCMAES(
                    None, 
                    problem, 
                    self.dim, 
                    budget= budget,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((self.dim,1)),
                    elitist = True
                ).run()
        
        if A2 == "Same":
            return
        
        if A2 == "Non-elitist":
            cma.parameters.elitist = False
            cma.parameters.budget = self.total_budget
        cma.run()
        
        
class Switched_From_CMA():
    def __init__(self, budget_factor, dim, A2, total_budget_factor=200):
        self.budget_factor = budget_factor
        self.dim = dim
        self.A2 = A2
        self.total_budget = total_budget_factor*self.dim
        
    def __call__(self, problem, A2, hparams = {}):
        
        cma = TrackedCMAES(
                    None, 
                    problem, 
                    self.dim, 
                    budget= self.dim * self.budget_factor,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((self.dim,1)),
                    elitist = True
                ).run()
        
        params = {}
        params['x_opt'] = cma.parameters.xopt
        params['pop'] = cma.parameters.population.x.T
        params['pop_f'] = cma.parameters.population.f
        params['stepsize'] = cma.parameters.sigma
        params['C'] = cma.parameters.C
        params['m'] = cma.parameters.m
        params['budget'] = self.total_budget
        
        
        algorithm = A2(problem, verbose = False, seed = np.random.get_state())
        # Set algorithm parameters based on parameters object
        algorithm.set_hyperparams(hparams)
        algorithm.set_params(params)
        
        def stopping_criteria():
            return problem.state.evaluations >= self.total_budget
        
        algorithm.set_stopping_criteria(stopping_criteria)
        algorithm.run()

        
def collect_A1_data(budget_factor, dim = 5):
    trigger = ioh.logger.trigger.Always()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'A1_data/A1_B{budget_factor*dim}_{dim}D',
        algorithm_name='ModCMA_A1',
        store_positions=True
    )
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
    
    for fid in range(1,25):
        for iid in range(1,6):
            problem = ioh.get_problem(fid, iid, dim, ProblemClass.BBOB)
            problem.attach_logger(logger)
            
            for rep in range(20):
                tracked_parameters.rep = rep
                tracked_parameters.iid = iid
                print(f"Running fundction {fid} instance {iid} repetition {rep} with A1, budget {budget_factor*dim}")
                np.random.seed(rep)
                cma = TrackedCMAES(
                    tracked_parameters, 
                    problem, 
                    dim, 
                    budget=dim * budget_factor,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((dim,1)),
                    elitist = True
                ).run()
                problem.reset()
            problem.detach_logger()
            
            
def collect_A2(budget_factor, dim, A2, algname):
    trigger = ioh.logger.trigger.OnImprovement()
    
    # logger = ioh.logger.Analyzer(
    #     triggers=[trigger],
    #     folder_name=f'A2_data_warm_MLSL/A2_{algname}_B{budget_factor*dim}_{dim}D',
    #     algorithm_name=algname,
    #     store_positions=False,
    # )
    tracked_parameters = TrackedParameters()
    # logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
    
    for fid in range(1,25):
        for iid in range(1,6):
            problem = ioh.get_problem(fid, iid, dim, ProblemClass.BBOB)
            # problem.attach_logger(logger)
            
            for rep in range(20):
                tracked_parameters.rep = rep
                tracked_parameters.iid = iid
                print(f"Running fundction {fid} instance {iid} repetition {rep} with A2 {algname}, budget {budget_factor*dim}")
                np.random.seed(rep)
                if algname in ["Same", "Non-elitist"]:
                    alg = From_CMA_To_CMA(budget_factor, dim, algname, total_budget_factor=200)
                    alg(problem, algname)
                else:
                    alg = Switched_From_CMA(budget_factor, dim, A2, total_budget_factor=200)
                    alg(problem, A2)
                problem.reset()
            problem.detach_logger()
            
            
def collect_all(x):
    budget_factor, dim = x
    for A2, algname in zip([MLSL, DE, PSO, BFGS, None, None], ["MLSL", "DE", "PSO", "BFGS", "Same", "Non-elitist"]):
        collect_A2(budget_factor, dim, A2, algname)
    # collect_A2(budget_factor, dim, BFGS, "BFGS")
                
                
def get_combinations():
    budget_factors = [10*i for i in range (1,21)] # 10, 20, ..., 1000
    dim = 5
    return [(bf, dim) for bf in budget_factors]
            
def process_ioh_data(base_path):
    dim = 5
    for budget_dir in os.listdir(base_path):
        if not (budget_dir == 'A1_B900_5D' or budget_dir == 'A1_B950_5D' or budget_dir == 'A1_B1000_5D'):
            continue
        budget_path = os.path.join(base_path, budget_dir)
        if not os.path.isdir(budget_path):
            continue

        all_rows = []

        for func_dir in os.listdir(budget_path):
            func_path = os.path.join(budget_path, func_dir)
            if not os.path.isdir(func_path):
                continue

            # Extract fid from directory name like 'data_f1_Sphere'
            try:
                fid = int(func_dir.split('_')[1][1:])
            except (IndexError, ValueError):
                print(f"Skipping malformed directory: {func_dir}")
                continue

            dat_file = os.path.join(func_path, f"IOHprofiler_f{fid}_DIM{dim}.dat")
            if not os.path.isfile(dat_file):
                continue

            try:
                df = pd.read_csv(dat_file, delim_whitespace=True, comment="#", dtype=str)
            except Exception as e:
                print(f"Error reading {dat_file}: {e}")
                continue

            # Filter out repeated header rows
            df = df[df['iid'] != 'iid']

            # Convert selected columns to numeric
            numeric_cols = ['evaluations', 'raw_y', 'rep', 'iid', 'x0', 'x1', 'x2', 'x3', 'x4']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Group by iid and compute true_y
            for iid_val, group in df.groupby('iid'):
                print(f"Processing fid={fid}, iid={iid_val}, budget dir={budget_dir}")
                try:
                    iid_int = int(float(iid_val))
                    problem = ioh.get_problem(fid, iid_int, dim, ProblemClass.BBOB)
                    optimum = problem.optimum.y
                except Exception as e:
                    print(f"Could not load problem fid={fid}, iid={iid_val}: {e}")
                    continue

                group = group[numeric_cols].copy()
                group['fid'] = fid
                group['true_y'] = group['raw_y'] + optimum
                all_rows.append(group)

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)

            # Reorder columns
            column_order = ['fid', 'iid', 'rep', 'evaluations', 'raw_y', 'true_y', 'x0', 'x1', 'x2', 'x3', 'x4']
            combined = combined[column_order]

            # Sort rows
            combined = combined.sort_values(by=['fid', 'iid', 'rep']).reset_index(drop=True)

            # Save CSV
            output_path = os.path.join(base_path, f"{budget_dir}.csv")
            combined.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

if __name__=='__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    
    x = get_combinations()
    temp = list(x)
    # for combination in temp:
    #     collect_all(combination
    partial_run = partial(collect_all)
    runParallelFunction(partial_run, temp)
