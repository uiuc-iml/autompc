import numpy as np

from .cost import Cost

class TrackingCost(Cost):
    """
    Time Varying Cost Wrapper
    """
    def __init__(self, system, cost, goal=None, **properties):
        """
        Create tracking cost cost.  Cost is:
        
            \sum_i cost(x[i],xg[i])
        
        where xg is a time series of goal state (may be None, in which case it is treated
        as zero).

        Parameters
        ----------
        system : System
            Robot system for which cost will be evaluated
        cost : Cost
            Cost that will be evaluated at every time step
        goal : numpy array of shape (reference trajectory length, self.obs_dim)
            Time seriese of goal state. Default is zero state
        properties : Dict
            a dictionary of properties that may be present in a cost and
            relevant to the selection of optimizers. Common values include:
            - 'goal': a time series of goal states (numpy array)
            - 'quad': whether the cost is quadratic (bool)
            - 'convex': whether the cost is convex (bool)
            - 'diff': whether the cost is differentiable (bool)
            - 'twice_diff': whether the cost is twice differentiable (bool)
        """
        super().__init__(system)
        self._cost = cost
        self.properties = {}
        if goal is None:
            goal = np.zeros((1, system.obs_dimi))
        self.goal = goal
        self.properties['goal'] = np.copy(goal)
        
        self.properties['quad'] = self._cost.is_quad
        self.properties['convex'] = self._cost.is_convex
        self.properties['diff'] = self._cost.is_diff
        self.properties['twice_diff'] = self._cost.is_twice_diff

    def incremental(self, obs, control, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.incremental(obs, control, t)

    def incremental_diff(self, obs, control, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.incremental_diff(obs, control, t) 

    def incremental_hess(self, obs, control, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.incremental_hess(obs, control, t)

    def terminal(self, obs, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.terminal(obs, t)

    def terminal_diff(self, obs, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.terminal_diff(obs, t)

    def terminal_hess(self, obs, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.terminal_hess(obs, t)

    @property
    def is_quad(self):
        """
        True if cost is quadratic.
        """
        return self._cost.properties.get('quad',False)

    @property
    def is_convex(self):
        """
        True if cost is convex.
        """
        return self._cost.properties.get('convex',False)

    @property
    def is_diff(self):
        """
        True if cost is differentiable.
        """
        return self._cost.properties.get('diff',False)

    @property
    def is_twice_diff(self):
        """
        True if cost is twice differentiable
        """
        return self._cost.properties.get('twice_diff', False)

    @property
    def has_goal(self):
        """
        True if cost has goal
        """
        return True
    
    def __add__(self, rhs):
        if isinstance(rhs,TrackingCost):
            if (self.goal is None and rhs.goal is None) or np.all(self.goal == rhs.goal):
                return TrackingCost(self.system,self._cost+rhs,self.goal)
        return Cost.__add__(self,rhs)

    def __mul__(self, rhs):
        if not isinstance(rhs,(float,int)):
            raise ValueError("* only supports product with numbers")
        return TrackingCost(self.system,self._cost*rhs,self.goal)
