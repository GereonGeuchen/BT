import numpy as np
# import random as rd
import math
import datetime
from shutil import make_archive
import copy

# My imports
from scipy.optimize import line_search
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import line_search
from scipy.optimize._linesearch import line_search_wolfe1
from scipy.optimize import line_search
from scipy.optimize import minimize

from algorithm import Algorithm

# My own functions

def line_search_wolfe12(f, grad, xk, pk, gfk=None, old_fval=None, old_old_fval=None,
                        c1=1e-4, c2=0.9, amin=1e-100, amax=1e100, bounds=None):
    """
    Tries More–Thuente Wolfe line search (line_search), then Wolfe1, then Armijo (BFGS-style).
    If bounds are provided, clamps evaluations to within those bounds.
    """
    print(vecnorm(gfk, ord=np.inf) )
    #Check if pk is descent direction
    print(np.dot(gfk, pk))
    if gfk is not None and vecnorm(gfk, ord=np.inf) <= 1e-5:
        return 0.0, 0, 0, f(xk), old_fval, gfk

    if bounds is not None:
        lb, ub = bounds

        def f_clipped(x):
            return f(np.clip(x, lb, ub))

        def grad_clipped(x):
            return grad(np.clip(x, lb, ub))
    else:
        f_clipped = f
        grad_clipped = grad

    # Try Wolfe2 (More–Thuente)
    try:
        res = line_search(
            f_clipped, grad_clipped, xk, pk, gfk, old_fval, old_old_fval,
            c1=c1, c2=c2, amax=amax
        )
        alpha_k, fc, gc, new_fval, new_old_fval, new_gfk = res
        if alpha_k is not None:
            return alpha_k, fc, gc, new_fval, new_old_fval, new_gfk
    except Exception:
        pass

    # Try Wolfe1 fallback
    try:
        res = line_search_wolfe1(
            f_clipped, grad_clipped, xk, pk, gfk, old_fval, old_old_fval,
            c1=c1, c2=c2, amin=amin, amax=amax
        )
        if res[0] is not None:
            return res
    except Exception:
        pass

    # Final fallback: Armijo-only (BFGS-style)
    try:
        res = line_search_BFGS(
            f_clipped, xk, pk, gfk, old_fval, c1=c1, alpha0=1.0
        )
        alpha_k, fc, gc, new_fval = res
        if alpha_k is not None:
            return alpha_k, fc, gc, new_fval, old_fval, None
    except Exception:
        pass
    print("Line search failed to find a suitable step size.")

def vecnorm(x, ord=None):
    """Calculate the vector norm of x."""
    if ord is None:
        return np.linalg.norm(x)
    elif ord == 1:
        return np.linalg.norm(x, ord=1)
    elif ord == 2:
        return np.linalg.norm(x, ord=2)
    elif ord == np.inf:
        return np.linalg.norm(x, ord=np.inf)
    else:
        raise ValueError(f"Unsupported norm order: {ord}")

class BFGS(Algorithm):

    """ Optimizer class for BFGS algorithm.

    Parameters:
    -----------------
    func : object, callable
        The function to be optimized.

    Attributes:
    -----------------

    gtol : float
        Value for gradient tolerance.

    norm : float

    eps : float

    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
        
    x0 : array-type
        The initial point for the algorithm routine.
        
    Hk : array-type
        Hessian approximation matrix at iteration k
        
    alpha_k : float
        Step size after iteration k
        
    Methods:
    --------------
    
    set_params(parameters):
        Sets algorithm parameters for warm-start.
        
    get_params(parameters):
        Transfers internal parameters to parameters dictionary.
        
    run():
        Runs the BFGS algorithm.
        
    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    def __init__(self, func, **kwargs):
        super(BFGS, self).__init__(func, **kwargs)
        self.gtol = 1e-12
        self.norm = np.inf
        self.eps = math.sqrt(np.finfo(float).eps)
        self.return_all = False
        self.jac = None
        self.finite_diff_rel_step = None
        self.x0 = None
        self.Hk = np.eye(self.dim, dtype=int)
        self.alpha_k = 0
        self.eval_count = []
        self.stepsizes = []
        self.matrix_norm = []
        self.Hk_overtime = []
        self.prec_overtime = []

    @classmethod
    def get_hyperparams(cls):
        return {'jac' : {'type' : 'c', 'range' : [None, '2-point', '3-point', 'cs']}}
    
    def set_params(self, parameters):
        self.budget = parameters.get('budget', 1)

        """Warm start routine"""

        # Initialize first point x0
        if 'x_opt' in parameters:
            self.x0 = parameters['x_opt']

        # Initialize stepsize alpha_k
            # some code

        # Initialize Hk
        if 'C' in parameters:
            self.Hk = parameters['C']


        #save number of evaluations and stepsizes to create plot
        
#         if 'evalcount' in parameters.internal_dict:
#             self.eval_count = parameters.internal_dict['evalcount']
            
#         if 'stepsizes' in parameters.internal_dict:
#             self.stepsizes = parameters.internal_dict['stepsizes']
        

    def get_params(self, parameters):
        parameters = super(BFGS, self).get_params(parameters)
        parameters['Hessian'] = self.Hk
        parameters['stepsize'] = self.alpha_k
#         parameters.internal_dict['x_opt'] = self.func.best_so_far_variables
#         parameters.internal_dict['x_hist'] = self.x_hist
#         parameters.internal_dict['f_hist'] = self.f_hist
        
#         parameters.internal_dict['evalcount'] = self.eval_count
        parameters['stepsizes'] = self.stepsizes
        parameters['matrix_norm'] = self.matrix_norm
#         parameters.internal_dict['a1_x_hist'] = self.x_hist.copy()
#         parameters.internal_dict['a1_f_hist'] = self.f_hist.copy()
#         parameters.internal_dict['BFGS_bestpoint'] = self.func.best_so_far_variables
#         parameters.internal_dict['evals_splitpoint'] = self.func.evaluations
        
        return parameters

    def gradient(self, x):
        """Calculate the gradient at point x using finite differences."""
        return approx_derivative(self.func, x, 
                                 rel_step=1e-8,
                                 bounds=(self.func.bounds.lb, self.func.bounds.ub))
    
    def run(self):
        """ Runs the BFGS algorithm.

        Parameters:
        --------------
        None

        Returns:
        --------------
        best_so_far_variables : array
                The best found solution.

        best_so_far_fvaluet: float
               The fitness value for the best found solution.

        """
        if self.verbose:
            print(f' BFGS started')
        
        print(self.Hk)

        return minimize(
            fun=self.func,
            x0=self.x0,
            method='BFGS',
            jac='3-point',
            hess='3-point',
            options={
                'maxiter': self.budget,
                'gtol': self.gtol,
                'norm': self.norm,
                'eps': self.eps,
                'return_all': self.return_all,
                'finite_diff_rel_step': self.finite_diff_rel_step,
                'hess_inv0': self.Hk
            },
            bounds=list(zip(self.func.bounds.lb, self.func.bounds.ub))
        )

        # Initialization 
        I = np.eye(self.dim, dtype=int)    # identity matrix
        k = 0

        # Initialize first point x0 at random
        if self.x0 is None:
            if self.uses_old_ioh:
                self.x0 = np.random.uniform(self.func.lowerbound, self.func.upperbound)
                
            else:
                self.x0 = np.random.uniform(self.func.bounds.lb, self.func.bounds.ub)
#             for i in range(0, self.dim):
#                 self.x0[i] = rd.uniform(-5, 5)

        # Prepare scalar function object and derive function and gradient function
#         def internal_func(x): #Needed since new functions return list by default
#             return self.func(x)[0]
        # sf = _prepare_scalar_function(self.func, self.x0, self.jac, epsilon=self.eps,
        #                       finite_diff_rel_step=self.finite_diff_rel_step)
        # f = sf.fun    # function object to evaluate function
        # gradient = sf.grad    # function object to evaluate gradient


        old_fval = self.func(self.x0)    # evaluate x0

        gfk = self.gradient(self.x0)   # gradient at x0
        
        self.x_hist.append(self.x0)
        self.f_hist.append(old_fval)

        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        xk = self.x0

        if self.return_all:
            allvecs = [self.x0]

        # Calculate initial gradient norm
        gnorm = vecnorm(gfk, ord = self.norm)
        
        # Algorithm loop
        while not self.stop():
            pk = -np.dot(self.Hk, gfk)    # derive direction pk from HK and gradient at x0 (gfk)
            print("Is Hk SPD? ", self.is_spd(self.Hk))
            """Derive alpha_k with Wolfe conditions.
            
            alpha_k : step size
            fc : count of function evaluations, gc: count of gradient evaluations
            old_fval : function value of new point xkp1 (xk + ak * pk)
            old_old_fval: function value of start point xk
            gfkp1 : gradient at new point xkp1
            """
            try:
                # self.alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                #      _line_search_wolfe12(self.func, self.gradient, xk, pk, gfk,
                #                           old_fval, old_old_fval, amin=1e-100, amax=1e100)
                self.alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                    line_search_wolfe12(
                        self.func, self.gradient, xk, pk, gfk,
                        old_fval, old_old_fval, c1=1e-4, c2=0.9,
                        amin=1e-100, amax=1e100,
                        bounds=(self.func.bounds.lb, self.func.bounds.ub)  
                    )
            except Exception as e:
                print(f"Line search failed with error: {e}")
                if self.verbose:
                    print(f"Line search failed with error: {e}")
                break

            
            # except _LineSearchError:
            #     if self.verbose:
            #         print('break because of line search error')
            #     break

            # Save parameters for plot and analysis
            self.stepsizes.append(self.alpha_k)
            self.matrix_norm.append(np.linalg.norm(self.Hk))
            self.Hk_overtime.append(self.Hk)
#             self.eval_count.append(self.func.evaluations)

            # calculate xk+1 with alpha_k and pk
            xkp1 = xk + self.alpha_k * pk

            # Make sure we sty within bounds
            xkp1 = np.clip(xkp1, self.func.bounds.lb, self.func.bounds.ub)
            if self.return_all:
                allvecs.append(xkp1)
            sk = xkp1 - xk    # step sk is difference between xk+1 and xk
            xk = xkp1    # make xk+1 new xk for next iteration
            self.x_hist.append(xk)
#             self.prec_overtime.append(self.func.best_so_far_precision)
            self.f_hist.append(old_fval)

            # Calculate gradient of xk+1 if not already found by Wolfe search
            if gfkp1 is None:
                gfkp1 = self.gradient(xkp1)
            yk = gfkp1 - gfk    # gradient difference
            gfk = gfkp1    # copy gradient to gfk for new iteration
            k += 1

            if not np.isfinite(old_fval):
                if self.verbose:
                    print('break because of np.isfinite')
                break

            # Check if gnorm is already smaller than tolerance
            # gnorm = vecnorm(gfk, ord=self.norm)
            # if (gnorm <= self.gtol):
            #     if self.verbose:
            #         print('break because of gnorm')
            #     break

            # Calculate rhok factor for inverse Hessian approximation matrix update
            print("Cross product: ", np.dot(yk, sk))
            try:
                rhok = 1.0 / (np.dot(yk, sk))
            except ZeroDivisionError:
                rhok = 1000.0
            if np.isinf(rhok):  # this is patch for NumPy
                rhok = 1000.0

            # Inverse Hessian approximation matrix Hk (Bk in papers) update
            A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
            A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
            self.Hk = np.dot(A1, np.dot(self.Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                     sk[np.newaxis, :])

        # Store found fitness value in fval for result
        fval = old_fval

        # Create OptimizeResult object based on found point and value
        # result = OptimizeResult(fun=fval, jac=gfk, hess_inv=self.Hk, nfev=sf.nfev,
        #                 njev=sf.ngev, x=xk,
        #                 nit=k)

        # if self.return_all:
        #     result['allvecs'] = allvecs
            
        if self.verbose:
            print(f'BFGS complete (stopping returned: {self.stop()})')
#             print(f'evals: {self.func.evaluations} x_opt: {self.func.best_so_far_variables}')

#         return self.func.best_so_far_variables, self.func.best_so_far_fvalue