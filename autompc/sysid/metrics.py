from typing import List, Optional
import numpy as np
from ..trajectory import Trajectory
from ..sysid.model import Model
from pdb import set_trace


def normalize(means, std, A):
    At = []
    for i in range(A.shape[1]):
        At.append((A[:,i] - means[i]) / std[i])
    return np.vstack(At).T

def get_model_residuals(model : Model, trajs : List[Trajectory], horizon=1) -> np.ndarray:
    resids = []
    for traj in trajs:
        state = np.array([model.traj_to_state(traj[:i+1]) for i in range(len(traj)-horizon)])
        for k in range(horizon):
            state = model.pred_batch(state, traj.ctrls[k:-(horizon-k), :])
        obs_traj = [model.get_obs(s) for s in state]
        actual = traj.obs[horizon:]
        resid = obs_traj - actual
        resids.append(resid)
    return np.vstack(resids)

def get_model_rmse(model : Model, trajs : List[Trajectory], horizon:int=1, dimension:Optional[int]=None) -> float:
    """
    Computes (unnormalized) RMSE at fixed horizon.

    By default, squared prediction errors are summed across observation
    dimensions.  If dimension is provided, then errors along a single
    observation dimension are extracted.

    Parameters
    ----------
    model : Model
        Model class to consider

    trajs : List of Trajectories
        Trajectories on which to evaluate

    horizon : int
        Prediction horizon at which to evaluate.
        Default is 1.
    
    dimension : int or None
        If None, sums squared errors across observation dimensions.
        If provided, extracts errors along a single observation dimension.
    """
    sqerrss = []
    for traj in trajs:
        state = np.array([model.traj_to_state(traj[:i+1]) for i in range(len(traj)-horizon)])
        for k in range(horizon):
            state = model.pred_batch(state, traj.ctrls[k:-(horizon-k), :])
        if dimension is None:
            obs_traj = np.array([model.get_obs(s) for s in state])
            actual = traj.obs[horizon:]
        else:
            obs_traj = np.array([model.get_obs(s)[dimension] for s in state])
            actual = traj.obs[horizon:,dimension]
        sqerrs = (obs_traj - actual) ** 2
        sqerrss.append(sqerrs)
    sqerrs = np.concatenate(sqerrss)
    if dimension is None:
        rmse = np.sqrt(np.mean(sqerrs, axis=None)*trajs[0].system.obs_dim)  #note: multiplication by obs_dim corrects for mean across observation dimensions 
    else:
        rmse = np.sqrt(np.mean(sqerrs, axis=None))
    return rmse

def get_model_abs_error(model : Model, trajs : List[Trajectory], horizon : int=1, dimension:Optional[int]=None) -> float:
    """
    Computes mean absolute error at fixed horizon

    By default, abs prediction errors are summed across observation
    dimensions.  If dimension is provided, then errors along a single
    observation dimension are extracted.

    Parameters
    ----------
    model : Model
        Model class to consider

    trajs : List of Trajectories
        Trajectories on which to evaluate

    horizon : int
        Prediction horizon at which to evaluate.
        Default is 1.
    
    dimension : int or None
        If None, sums abs errors across observation dimensions.
        If provided, extract errors along a single observation dimension.
    """

    abserrss = []
    for traj in trajs:
        state = np.array([model.traj_to_state(traj[:i+1]) for i in range(len(traj)-horizon)])
        for k in range(horizon):
            state = model.pred_batch(state, traj.ctrls[k:-(horizon-k), :])
        if dimension is None:
            obs_traj = [model.get_obs(s) for s in state]
            actual = traj.obs[horizon:]
        else:
            obs_traj = [model.get_obs(s)[dimension] for s in state]
            actual = traj.obs[horizon:,dimension]
        abserrs = np.abs(obs_traj - actual)
        abserrss.append(abserrs)
    absolute_errors = np.concatenate(abserrss)
    if dimension is None:
        mae = np.mean(absolute_errors, axis=None)*trajs[0].system.obs_dim  #note: multiplication by obs_dim corrects for mean across observation dimensions
    else:
        mae = np.mean(absolute_errors, axis=None)
    return mae


def get_model_rmsmens(model : Model, trajs : List[Trajectory], horiz : int=1, dimension:Optional[int]=None) -> float:
    R"""
    Compute root mean squared model error, normalized step-wise (RMSMENS).

    Given a set of trajectories :math:`\mathcal{T}`, let :math:`x_{i,t} \in \mathbb{R}^n, u_{i,t} \in \mathbb{R}^m` denote the state and control input in the :math:`i^\textrm{th}` trajectory at time :math:`t`. 
    Let :math:`L_i` denote the length of the :math:`i\textrm{th}` trajectory.  Given a predictive model :math:`g`, let :math:`g(i,t,k)` give the prediction for :math:`x_{i,t+k}` given the 
    states :math:`\mathbf{x}_{i,1:t}` and controls :math:`\mathbf{u}_{i,1:t+k-1}`.
    Let

    .. math::
        \sigma = \textrm{std} \{ x_{i,t+1} - x_{i,t} \mid i=1,\ldots,\left|\mathcal{T}\right|
                        t=1,\ldots,L_i-1 \}
        \textrm{,}

    where the mean and standard deviation are computed element-wise.

    Denote the :math:`k`-step Root Mean Squared Model Error, Normalized Step-wise, RMSMENS(:math:`k`), of :math:`g` with respect to :math:`\mathcal{T}` as

    .. math::

      \sqrt{
       \frac{1}{n}
        \left\lVert
          \frac{1}{\left|\mathcal{T}\right|}
          \sum_{i=1}^{\left|\mathcal{T}\right|}
          \frac{1}{L_i-k  }
          \sum_{t=1}^{L_i-k}\frac{1}{\sigma^2} e(i,t,k)^2
        \right\rVert_1
      }
      \textrm{,}

    where

    .. math::
      e(i,t,k) = g(i,t,k) - g(i,t,k-1) - (x_{i,t+k} - x_{i,t+k-1})

    By default, squared prediction errors are summed across observation
    dimensions.  If dimension is provided, then errors along a single
    observation dimension are extracted.

    Parameters
    ----------
    model : Model
        Model class to consider

    trajs : List of Trajectories
        Trajectories on which to evaluate

    horizon : int
        Prediction horizon at which to evaluate.
        Default is 1.
    
        
    dimension : int or None
        If None, sums squared errors across observation dimensions.
        If provided, extracts errors along a single observation dimension.
    """
    if dimension is None:
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
    else:
        dY = np.concatenate([traj.obs[1:,dimension] - traj.obs[:-1,dimension] for traj in trajs])
    dy_means = np.mean(dY, axis=0)
    dy_std = np.std(dY, axis=0)

    sqerrss = []
    for traj in trajs:
        state = np.array([model.traj_to_state(traj[:i+1]) for i in range(len(traj)-horiz)])
        for k in range(horiz):
            pstate = state
            state = model.pred_batch(state, traj.ctrls[k:-(horiz-k), :])
        pred_deltas = state - pstate
        if dimension is None:
            obs_deltas = [model.get_obs(s) for s in pred_deltas]
            act_deltas = traj.obs[horiz:] - traj.obs[horiz-1:-1]
        else:
            obs_deltas = [model.get_obs(s)[dimension:dimension+1] for s in pred_deltas]
            act_deltas = traj.obs[horiz:,dimension:dimension+1] - traj.obs[horiz-1:-1,dimension:dimension+1]
        norm_pred_deltas = normalize(dy_means, dy_std, obs_deltas)
        norm_act_deltas = normalize(dy_means, dy_std, act_deltas)
        sqerrs = (norm_pred_deltas - norm_act_deltas) ** 2
        sqerrss.append(sqerrs)
    sqerrs = np.concatenate(sqerrss)
    rmse = np.sqrt(np.mean(sqerrs, axis=None))
    return rmse
