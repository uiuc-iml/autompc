# Standard library includes
import sys, time

# External library includes
import numpy as np
import mujoco_py

# Project includes
from .benchmark import Benchmark
from ..utils.data_generation import *
from .. import System
from ..task import Task
from ..trajectory import Trajectory
from ..costs import Cost, QuadCost

def mujoco_dynamics(env, x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):]
    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
            old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()
    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel

    return np.concatenate([new_qpos, new_qvel])

#TODO: Cost is used for controller tuning.
#Different data may have different costs. 
class MujocoCost(Cost):
    def __init__(self, env):
        super().__init__(self, None)
        self.env = env
    
    def __call__(self, traj):
        cum_reward = 0.0
        for i in range(len(traj)-1):
            reward_ctrl = -0.1 * np.square(traj[i].ctrl).sum()
            reward_run = (traj[i+1, "x0"] - traj[i, "x0"]) / self.env.dt
            cum_reward += reward_ctrl + reward_run
        return 200 - cum_reward
    
    def incremental(self,obs,ctrl):
        raise NotImplementedError

    def terminal(self,obs):
        raise NotImplementedError

def gen_trajs(env, system, num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    env.seed(int(rng.integers(1 << 30)))
    env.action_space.seed(int(rng.integers(1 << 30)))
    for i in range(num_trajs):
        init_obs = env.reset()
        traj = Trajectory.zeros(system, traj_len)
        traj[0].obs[:] = np.concatenate([[0], init_obs])
        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = mujoco_dynamics(env, traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs

class OpenAIBenchmark(Benchmark):
    def __init__(self, name, data_gen_method="uniform_random"):
        
        import gym, mujoco_py
        env = gym.make(name)
        self.env = env
        system = ampc.System([f"x{i}" for i in range(18)], [f"u{i}" for i in range(6)], env.dt)

        system.dt = env.dt
        task = Task(system)

        # cost = MujocoCost(env)
        # task = Task(system,cost)
        # task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
        # init_obs = np.concatenate([env.init_qpos, env.init_qvel])
        # task.set_init_obs(init_obs)
        # task.set_num_steps(200)

        # factory = QuadCost(system, goal = np.zeros(system.obs_dim))
        # for obs in system.observations:
        #     if not obs in ["x1", "x6", "x7", "x8", "x9"]:
        #         factory.fix_Q_value(obs, 0.0)
        #     if not obs in ["x1", "x9"]:
        #         factory.fix_F_value(obs, 0.0)
        # factory.set_tunable_goal("x9", lower_bound=0.0, upper_bound=5.0, default=1.0)
        # self.cost_factory = factory

        super().__init__(name, system, task, data_gen_method)

    def dynamics(self, x, u):
        return mujoco_dynamics(self.env,x,u)
    
    def gen_trajs(self, seed, n_trajs, traj_len=200):
        return gen_trajs(self.env, self.system, n_trajs, traj_len, seed)
    
    @staticmethod
    def data_gen_methods():
        return ["uniform_random"]