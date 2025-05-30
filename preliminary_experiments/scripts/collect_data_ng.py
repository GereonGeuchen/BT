import os 
import shutil
import argparse
from dataclasses import dataclass, fields

import ioh
from modcma import ModularCMAES, Parameters
import numpy as np

from dynas.bfgs import BFGS
from dynas.pso import PSO
from dynas.mlsl import MLSL
from dynas.de import DE
import warnings
from itertools import product
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
from nevergrad import functions
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
    def __init__(self, budget_factor, dim, A2, total_budget_factor=300):
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
    def __init__(self, budget_factor, dim, A2, total_budget_factor=300):
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
        folder_name=f'A1_B{budget_factor}_{dim}D',
        algorithm_name='ModCMA_A1',
        store_positions=True,
    )
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
    
    for fid in range(25,46):
        # for iid in range(1,11):
        problem = ioh.get_problem(fid, 0, dim, problem_type='BBOB')
        problem.attach_logger(logger)

        for rep in range(50):
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
    
    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'A2_{algname}_B{budget_factor}_{dim}D',
        algorithm_name=algname,
        store_positions=False,
    )
    # tracked_parameters = TrackedParameters()
    # logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
    
    for fid in range(25,46):
        # for iid in range(1,11):
        problem = ioh.get_problem(fid, 0, dim, problem_type='BBOB')
        problem.attach_logger(logger)

        for rep in range(50):
            np.random.seed(rep)
            if algname in ["Same", "Non-elitist"]:
                alg = From_CMA_To_CMA(budget_factor, dim, algname, total_budget_factor=300)
                alg(problem, algname)
            else:
                alg = Switched_From_CMA(budget_factor, dim, A2, total_budget_factor=300)
                alg(problem, A2)
            problem.reset()
        problem.detach_logger()
            
            
def collect_all(x):
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.hm(np.array(x)), 'ng_hm', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.rastrigin(np.array(x)), 'ng_rastrigin', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.griewank(np.array(x)), 'ng_griewank', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.rosenbrock(np.array(x)), 'ng_rosenbrock', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.ackley(np.array(x)), 'ng_ackley', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.lunacek(np.array(x)), 'ng_lunacek', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.deceptivemultimodal(np.array(x)), 'ng_deceptivemultimodal', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.bucherastrigin(np.array(x)), 'ng_bucherastrigin', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.multipeak(np.array(x)), 'ng_multipeak', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.sphere(np.array(x)), 'ng_sphere', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.doublelinearslope(np.array(x)), 'ng_doublelinearslope', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    def sdls(x):
        try:
            res = functions.corefuncs.stepdoublelinearslope(np.array(x))
        except:
            res = np.inf
        return res
    ioh.problem.wrap_real_problem(sdls, 'ng_stepdoublelinearslope', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.cigar(np.array(x)), 'ng_cigar', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.altcigar(np.array(x)), 'ng_altcigar', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.ellipsoid(np.array(x)), 'ng_ellipsoid', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.altellipsoid(np.array(x)), 'ng_altellipsoid', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.stepellipsoid(np.array(x)), 'ng_stepellipsoid', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.discus(np.array(x)), 'ng_discus', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.bentcigar(np.array(x)), 'ng_bentcigar', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.deceptiveillcond(np.array(x)), 'ng_deceptiveillcond', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    ioh.problem.wrap_real_problem(lambda x : functions.corefuncs.deceptivepath(np.array(x)), 'ng_deceptivepath', optimization_type=ioh.OptimizationType.Minimization, lb=-5.0, ub=5.0)
    budget_factor, dim = x
    collect_A1_data(budget_factor, dim)
    for A2, algname in zip([MLSL, DE, PSO, BFGS, None, None], ["MLSL", "DE", "PSO", "BFGS", "Same", "Non-elitist"]):
        collect_A2(budget_factor, dim, A2, algname)
                
                
def get_combinations():
    budget_factors = [30,60]
    dims = [5,10]
    return product(budget_factors, dims)                

if __name__=='__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    
    x = get_combinations()
    temp = list(x)

    partial_run = partial(collect_all)
    runParallelFunction(partial_run, temp)