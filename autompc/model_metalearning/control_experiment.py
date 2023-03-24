from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np

from .. import Controller
from ..tuning import ControlTuner
from ..sysid import MLP, AutoSelectModel
from ..sysid.model import FixedModel
from ..ocp import QuadCostTransformer
from ..optim import IterativeLQR
from ..benchmarks.meta_benchmarks.gym_mujoco import GymExtensionBenchmark
from ..benchmarks import CartpoleSwingupBenchmark

from meta_utils import load_data, load_cfg

DATA_PATH = Path(__file__).parent / "meta_data"
CFG_PATH = Path(__file__).parent / "meta_cfg"

def get_controller(benchmark, sysid_trajs, model, transformer):
    # Create controller
    controller = Controller(benchmark.system)
    controller.set_ocp_transformer(transformer)
    controller.set_optimizer(IterativeLQR(benchmark.system))
    controller.set_model(model)
    controller.set_ocp(benchmark.task)

    controller.build(sysid_trajs)

    return controller

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

def run_experiment(args):
    with open(args.cfg1, "rb") as f:
        config_1 = pickle.load(f)
    with open(args.cfg2, "rb") as f:
        config_2 = pickle.load(f)
    
    if args.system_name == "CartpoleSwingup":
        benchmark = CartpoleSwingupBenchmark()
        system = benchmark.system
        trajs = benchmark.gen_trajs(n_trajs=500, traj_len=200, seed=0)
        transformer = QuadCostTransformer(benchmark.system)
        transformer.fix_R_value("u", 0.01)
    else:
        system, trajs = load_data(path=DATA_PATH, name=args.system_name)
        benchmark = GymExtensionBenchmark(name=args.system_name)
        transformer = get_ocp_transformer(system)
    
    model_1 = AutoSelectModel(benchmark.system)
    model_2 = AutoSelectModel(benchmark.system)
    model_1.set_config(config_1)
    model_2.set_config(config_2)

    controller_1 = get_controller(benchmark, trajs, model_1, transformer)
    controller_2 = get_controller(benchmark, trajs, model_2, transformer)

    traj_1, cost_1, termcond_1 = benchmark.task.simulate(controller_1, benchmark.dynamics)
    traj_2, cost_2, termcond_2 = benchmark.task.simulate(controller_2, benchmark.dynamics)

    print(f"{cost_1=}")
    print(f"{cost_2=}")

    breakpoint()




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--system_name", "-s")
    parser.add_argument("--cfg1")
    parser.add_argument("--cfg2")
    args = parser.parse_args()

    run_experiment(args)
