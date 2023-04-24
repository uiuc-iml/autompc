from argparse import ArgumentParser
from pathlib import Path
import pickle, json
from collections import namedtuple

import numpy as np
from ConfigSpace import Configuration

from .. import Controller
from ..tuning import ControlTuner
from ..sysid import MLP, AutoSelectModel
from ..sysid.model import FixedModel
from ..ocp import QuadCostTransformer
from ..optim import IterativeLQR
from ..benchmarks.meta_benchmarks.gym_mujoco import GymExtensionBenchmark
from ..benchmarks import CartpoleSwingupBenchmark

from meta_utils import load_data, load_cfg

CONFIG_PATH = "controller_cfgs/"
ExperimentLabel = namedtuple("ExperimentLabel", "portfolio_size seed")

def get_ocp_transformer(system):
    # Get index of x velocity
    x_vel_index = int(len(system.observations) / 2)

    # Allow transformer to tune target x-velocity
    transformer = QuadCostTransformer(system, goal=np.zeros(system.obs_dim))
    transformer.set_tunable_goal(f"x{x_vel_index}", lower_bound=0.0, upper_bound=5.0, default=1.0)

    # Fix all other Q and F values to 0
    for i in range(system.obs_dim):
        if i != x_vel_index:
            transformer.fix_Q_value(f"x{i}", 0)
            transformer.fix_F_value(f"x{i}", 0)

    return transformer

def get_controller(benchmark, transformer, trajs, cfg_dict):
    model = MLP(benchmark.system)
    controller = Controller(benchmark.system)
    controller.set_ocp_transformer(transformer)
    controller.add_model(model)
    optim = IterativeLQR(benchmark.system)
    controller.add_optimizer(optim)

    cs = controller.get_config_space()
    cfg = Configuration(cs, values=cfg_dict)
    controller.set_config(cfg)

    controller.set_ocp(benchmark.task)
    controller.build(trajs)

    return controller

def run_experiment(benchmark, config):
    system = benchmark.system
    trajs = benchmark.gen_trajs(n_trajs=100, traj_len=200, seed=100)
    transformer = QuadCostTransformer(benchmark.system)
    
    controller = get_controller(benchmark, transformer, trajs, config)

    traj, cost, termcond = benchmark.task.simulate(controller, benchmark.dynamics)

    return cost, traj

def main():
    benchmark = CartpoleSwingupBenchmark()
    benchmark_name = "CartpoleSwingup"
    portfolio_sizes = [0,5,10]
    seeds = [0,1,2,3,4]

    costs = dict()
    trajs = dict()

    # Run experiments
    for portfolio_size in portfolio_sizes:
        for seed in seeds:
            config_path = Path(CONFIG_PATH) / f"{benchmark_name}_port_{portfolio_size}_seed_{seed}.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            cost, traj = run_experiment(benchmark, config)

            experiment_label = ExperimentLabel(portfolio_size=portfolio_size, seed=seed)
            costs[experiment_label] = cost
            trajs[experiment_label] = traj

    # Compute summary statistics and print
    print(f"Results for {benchmark_name}")
    print(f"=================================")
    for portfolio_size in portfolio_sizes:
        costs_for_size = [costs[ExperimentLabel(portfolio_size=portfolio_size, seed=seed)] for seed in seeds]
        print(f"Portfolio Size: {portfolio_size}, Mean: {np.mean(costs_for_size):.4f}, Std: {np.std(costs_for_size):.4f}, Raw Values: {costs_for_size}")

    breakpoint()

if __name__ == "__main__":
    main()