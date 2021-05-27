from collections import Iterable
from .controller import Controller, ControllerFactory
from ..constraint import Constraint
from ..hyper import IntRangeHyperparam
from ..cost import Cost
from pdb import set_trace
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import cvxpy as cp
import numpy as np

from pyoptsolver import OptProblem, OptConfig, OptSolver


class TrajOptProblem(object):
    """Just a general interface for nonlinear optimization problems.
    I will just use knitro/ipopt style and the snopt one is easily written as well.

    Args:
        nx (int): dimension of the decision variable
        nc (int): dimension of the constraints
    """
    def __init__(self, nx, nc):
        self.dimx = nx
        self.dimc = nc
        self.xlb, self.xub = np.zeros((2, nx))
        self.clb, self.cub = np.zeros((2, nc))

    def get_cost(self, x):
        raise NotImplementedError("Sub-class has to implement get_cost function.")

    def get_gradient(self, x):
        raise NotImplementedError("Sub-class has to implement get_gradient function.")

    def get_constraint(self, x):
        raise NotImplementedError("Sub-class has to implement get_constraint function.")

    def get_jacobian(self, x, return_rowcol):
        """This function computes the Jacobian at current solution x, if return_rowcol is True, it has to return row and col, too."""
        raise NotImplementedError("Sub-class has to implement get_jacobian function.")


class BoundedConstraint(object):
    """This class implements constraints that is like lb <= c <= ub"""
    def __init__(self, dim, lb, ub):
        self.dim = dim
        self.lb = lb
        self.ub = ub

    def eval(self, xs, ret_grad=False):
        raise NotImplementedError('Subclass has to implement eval function')


class PathConstraint(BoundedConstraint):
    def __init__(self, dim, lb, ub):
        BoundedConstraint.__init__(self, dim, lb, ub)

    def eval(self, x, u, ret_grad=False):
        """For path constraint, both x and u are provided. For terminal, u is None and ret_grad returns one jacobian only"""
        raise NotImplementedError("Subclass has to implement eval function.")


class ConstrContainer(object):
    """This container include information of constraints.
    Basically, the user can insert various types of constraint into this container.
    During insertion, they have to specify which point this constraint is evaluated.
    """
    def __init__(self):
        self.terminal_constrs = []
        self.point_constrs = []

    def add_terminal(self, constr):
        self.terminal_constrs.append(constr)
    
    def add_point(self, index, constr):
        assert index >= 0
        self.point_constrs.append((index, constr))

    def compute_dim(self, horizon):
        """Compute dimension of all constraints, it depends on horizon"""
        dim1 = sum([tc.dim for tc in self.terminal_constrs])
        dim2 = sum([pc.dim * (horizon if index is None else 1) for index, pc in self.point_constrs])
        return dim1 + dim2


class NonLinearMPCProblem(TrajOptProblem):
    """Just write the NonLinear MPC problem in the OptProblem style.
    """
    def __init__(self, system, model, task, horizon):
        assert task.get_cost().is_diff()
        self.system = system
        self.task = task
        self.model = model
        self.horizon = horizon
        dc = system.ctrl_dim
        ds = system.obs_dim
        self.ctrl_dim = dc
        self.obs_dim = ds
        # now I can get the size of the problem
        eq_cons = task.get_eq_constraints()
        ineq_cons = task.get_eq_constraints()
        nx = ds * (horizon + 1) + dc * horizon  # x0 to xN, u0 to u_{N-1}
        nf = horizon * (eq_cons.dim + ineq_cons.dim) + horizon * ds  # for dynamics and other constraints
        TrajOptProblem.__init__(self, nx, nf)
        self._create_cache()

    def _create_cache(self):
        self._x = np.zeros(self.dimx)
        self._grad = np.zeros(self.dimx)
        self._c = np.zeros(self.dimc)
        self._c_dyn = self._c[-self.horizon * self.obs_dim:].reshape((self.horizon, -1))  # the last parts store dynamics
        len1 = (self.horizon + 1) * self.obs_dim
        len2 = self.horizon * self.ctrl_dim
        self._state = self._x[:len1].reshape((self.horizon + 1, self.obs_dim))
        self._ctrl = self._x[len1:].reshape((self.horizon, self.ctrl_dim))
        self._grad_state = self._grad[:len1].reshape((self.horizon + 1, self.obs_dim))
        self._grad_ctrl = self._grad[len1:].reshape((self.horizon, self.ctrl_dim))
        self._x[:] = np.random.random(self.dimx)
        self._row, self._col = self.get_jacobian(self._x, True)
        self._jac = np.zeros(self._row.size)
    
    @property
    def nnz(self):
        return self._jac.size

    def get_cost(self, x):
        # compute the cost function, not sure how it's gonna be written though
        cost = self.task.get_cost()
        self._x[:] = x  # copy contents in
        dt = self.system.dt
        tc = cost.eval_term_obs_cost(self._state[-1])
        for i in range(self.horizon + 1):
            tc += cost.eval_obs_cost(self._state[i]) * dt
        for i in range(self.horizon):
            tc += cost.eval_ctrl_cost(self._ctrl[i]) * dt
        return tc

    def get_gradient(self, x):
        """Compute the gradient given some guess"""
        self._x[:] = x
        self._grad[:] = 0  # reset just in case
        # terminal one
        cost = self.task.get_cost()
        _, gradtc = cost.eval_term_obs_cost_diff(self._state[-1])
        self._grad_state[-1] = gradtc
        dt = self.system.dt
        for i in range(self.horizon + 1):
            _, gradx = cost.eval_obs_cost_diff(self._state[i])
            self._grad_state[i] += gradx * dt
        for i in range(self.horizon):
            _, gradu = cost.eval_ctrl_cost_diff(self._ctrl[i])
            self._grad_ctrl[i] = gradu * dt
        return self._grad

    def get_constraint(self, x):
        """Evaluate the constraint function"""
        self._x[:] = x
        self._c[:] = 0
        # first compute for dynamics
        pred_states = self.model.pred_parallel(self._state[:self.horizon], self._ctrl[:self.horizon])
        for i in range(self.horizon):
            self._c_dyn[i] = -self._state[i + 1] + pred_states[i]
        # then path constraints
        cr = 0  # means currow
        eq_cons = self.task.get_eq_constraints()
        ineq_cons = self.task.get_ineq_constraints()
        for i in range(self.horizon):
            v, j = eq_cons.eval_diff(self._state[1 + i])
            self._c[cr: cr + v.size] = v
            cr += v.size
            v2, j2 = ineq_cons.eval_diff(self._state[1 + i])
            self._c[cr: cr + v2.size] = v2
            cr += v2.size
        # currently we do not have terminal constraint so let it be, we will come back later
        return self._c

    def get_constr_bounds(self):
        """Just return the bounds of constraints"""
        clb, cub = np.zeros((2, self.dimc))
        # start from terminal_constrs
        cr = 0
        eq_cons = self.task.get_eq_constraints()
        ineq_cons = self.task.get_eq_constraints()
        for i in range(self.horizon):
            # v, j = self.task.eval_diff_eq_cons(obs[1 + i])
            # self._c[cr: cr + v.size] = v
            cr += eq_cons.dim
            # v2, j2 = self.task.eval_diff_ineq_cons(obs[1 + i])
            crv = cr + ineq_cons.dim
            clb[cr: crv] = -1e10
            cr = crv
        return clb, cub

    def get_variable_bounds(self):
        obsbd = self.task.get_obs_bounds()
        ctrlbd = self.task.get_ctrl_bounds()
        dc = self.ctrl_dim
        ds = self.obs_dim
        xlb, xub = np.zeros((2, self.dimx))
        xlb[:(self.horizon + 1) * ds].reshape((-1, ds))[:] = obsbd[:, 0]
        xub[:(self.horizon + 1) * ds].reshape((-1, ds))[:] = obsbd[:, 1]
        xlb[-self.horizon * dc:].reshape((-1, dc))[:] = ctrlbd[:, 0]
        xub[-self.horizon * dc:].reshape((-1, dc))[:] = ctrlbd[:, 1]
        return xlb, xub

    def _dense_to_rowcol(self, shape, row0, col0):
        row, col = shape
        rows = np.arange(row)[:, None] * np.ones(col) + row0
        cols = np.ones((row, 1)) * np.arange(col) + col0
        return rows.flatten(), cols.flatten()

    def get_state_index(self, index):
        return index * self.obs_dim

    def get_ctrl_index(self, index):
        return (self.horizon + 1) * self.obs_dim + index * self.ctrl_dim

    def get_jacobian(self, x, return_rowcol):
        """This function computes the Jacobian at current solution x, if return_rowcol is True, it returns a tuple of the patterns of row and col"""
        self._x[:] = x
        # Here I may as well assume all the  ret_grad stuff returns a dense jacobian or None which means all zero, support for coo_matrix is under development
        dims = self.obs_dim
        dimu = self.ctrl_dim
        if return_rowcol:
            cr = 0
            row = []
            col = []
            # just routinely evalute equality and inequality constraints
            eq_cons = self.task.get_eq_constraints()
            ineq_cons = self.task.get_eq_constraints()
            rowjac1, coljac1 = self._dense_to_rowcol((eq_cons.dim, dims), 0, 0)
            rowjac2, coljac2 = self._dense_to_rowcol((ineq_cons.dim, dims), 0, 0)
            base_x_idx = dims  # starts from the second point...
            for i in range(self.horizon):
                row.append(rowjac1 + cr)
                col.append(coljac1 + base_x_idx + i * dims)
                cr += eq_cons.dim
                row.append(rowjac2 + cr)
                col.append(coljac2 + base_x_idx + i * dims)
                cr += ineq_cons.dim
            # finally for dynamics
            _, mat1, mat2 = self.model.pred_diff(self._state[0], self._ctrl[0])
            srowptn, scolptn = self._dense_to_rowcol(mat1.shape, 0, 0)
            urowptn, ucolptn = self._dense_to_rowcol(mat2.shape, 0, 0)
            # compute patterns for it
            base_x_idx = 0
            base_u_idx = dims * (self.horizon + 1)
            for i in range(self.horizon):
                row.append(cr + srowptn)
                col.append(base_x_idx + i * dims + scolptn)
                row.append(cr + urowptn)
                col.append(base_u_idx + i * dimu + ucolptn)
                # take care, here you are placing them after placing jacobian
                row.append(cr + np.arange(dims))
                col.append(base_x_idx + (i + 1) * dims + np.arange(dims))
                cr += dims
            return np.concatenate(row), np.concatenate(col)
        else:
            # I have to compute the jacobian here
            cr = 0
            cg = 0
            self._jac[:] = 0
            # for terminal constraints first
            ###### Placeholder for terminal constraints
            # then other point constraints
            eq_cons = self.task.get_eq_constraints()
            ineq_cons = self.task.get_ineq_constraints()
            for i in range(self.horizon):
                v, j = eq_cons.eval_diff(self._state[1 + i])
                self._jac[cg: cg + j.size] = j.flat
                cg += j.size
                v2, j2 = ineq_cons.eval_diff(self._state[1 + i])
                self._jac[cg: cg + j2.size] = j2.flat
                cg += j2.size
            # finally for dynamics
            _, matss, matus = self.model.pred_diff_parallel(self._state[:self.horizon], self._ctrl[:self.horizon])
            for i in range(self.horizon):
                mats, matu = matss[i], matus[i]
                self._jac[cg: cg + mats.size] = mats.flat
                cg += mats.size
                self._jac[cg: cg + matu.size] = matu.flat
                cg += matu.size
                self._jac[cg: cg + dims] = -1
                cg += dims
            return self._jac


class IpoptWrapper(OptProblem):
    """Just the ipopt style stuff"""
    def __init__(self, prob):
        assert isinstance(prob, TrajOptProblem)
        self.prob = prob
        OptProblem.__init__(self, prob.dimx, prob.dimc, prob.nnz)
        self.get_lb()[:], self.get_ub()[:] = prob.get_constr_bounds()
        self.get_xlb()[:], self.get_xub()[:] = prob.get_variable_bounds()
        self.ipopt_style()

    def __cost__(self, x):
        return self.prob.get_cost(x)

    def __gradient__(self, x, y):
        y[:] = self.prob.get_gradient(x)
        return True

    def __constraint__(self, x, y):
        y[:] = self.prob.get_constraint(x)
        return 0

    def __jacobian__(self, x, jac, row, col, rec):
        if rec:
            row_, col_ = self.prob.get_jacobian(x, True)
            row[:] = row_
            col[:] = col_
        else:
            jac[:] = self.prob.get_jacobian(x, False)
        return 0

class DirectTranscriptionControllerFactory(ControllerFactory):
    """
    Direct Transcription (DT) is a method to discretize an optimal control problem which is inherently continuous.
    Such discretization is usually necessary in order to get an optimization problem of finite dimensionality.
    For a trajectory with time length :math:`T`_, it discretize the time interval into a equidistant grid of size :math:`N`, called knots.
    The state and control at each knot are optimized.
    The constraints are imposed at the knots, including system dynamics constraints.
    DT uses first-order Euler integration to approximate the constraints of system dynamics.
    The details can be found in `An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation <https://epubs.siam.org/doi/pdf/10.1137/16M1062569>`_.

    Hyperparameter:
    - *horizon* (Type: int, Lower: 1, High: 30, Default: 10): Control Horizon
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.Controller = DirectTranscriptionController
        self.name = "DirectTranscription"

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        horizon = CSH.UniformIntegerHyperparameter(name="horizon",
                lower=1, upper=30, default_value=10)
        cs.add_hyperparameter(horizon)
        return cs

class DirectTranscriptionController(Controller):
    """
    Implementation of the linear controller. For this very basic version, it accepts some linear models and compute output.
    constraints is a dict of constraints we have to consider, it has two keys: path and terminal. The items are list of Constraints.
    cost is a Cost instance to compute fitness of a trajectory
    """
    def __init__(self, system, model, task, horizon):
        # I prefer type checking, but clearly current API does not allow me so
        Controller.__init__(self, system, model, task)
        self.horizon = int(np.ceil(horizon / system.dt))
        self._built = False
        self._guess = None
        self._x_dim = (self.horizon + 1) * system.obs_dim + self.horizon * system.ctrl_dim

    def set_guess(self, guess):
        if guess.size != self._xdim:
            raise Exception("Guess dimension should be %d" % self._x_dim)
        self._guess = guess

    def _build_problem(self):
        """Use cvxpy to construct the problem"""
        self._built = True
        self.problem = NonLinearMPCProblem(self.system, self.model, self.task, self.horizon)
        self.wrapper = IpoptWrapper(self.problem)

    def _update_problem_and_solve(self, x0):
        """Solve the problem"""
        if not self._built:
            self._build_problem()
        dims = self.problem.obs_dim
        self.wrapper.get_xlb()[:dims] = self.wrapper.get_xub()[:dims] = x0  # so I set this one
        config = OptConfig(backend='ipopt', print_level=5, opt_tol=1e-3,
                max_iter=10)
        solver = OptSolver(self.wrapper, config)
        if self._guess is None:
            rst = solver.solve_rand()
        else:
            rst = solver.solve_guess(self._guess)
        return rst

    @property
    def state_dim(self):
        return self.model.state_dim+self.model.ctrl_dim

    @staticmethod
    def is_compatible(system, task, model):
        #TODO: this part is really confusing...
        return True  # this should be universal...
 
    def traj_to_state(self, traj):
        return np.concatenate([self.model.traj_to_state(traj),
                traj[-1].ctrl])

    def run(self, state, new_obs):
        x = self.model.update_state(state[:-self.system.ctrl_dim],
                state[-self.system.ctrl_dim:], new_obs)
        self._x_cache = x
        print('state is ', x)
        rst = self._update_problem_and_solve(x)
        print(rst.flag)
        sol = rst.sol.copy()
        # update guess
        self._guess = sol
        dims = self.problem.obs_dim
        dimu = self.problem.ctrl_dim
        idx0 = dims * (self.horizon + 1)
        # print('path is ', sol[:idx0].reshape((-1, dims)))
        u = sol[idx0: idx0 + dimu]
        statenew = np.concatenate([x, u])
        return u, statenew