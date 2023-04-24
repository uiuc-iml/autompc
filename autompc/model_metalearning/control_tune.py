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

from meta_utils import load_data, load_cfg

DATA_PATH = Path(__file__).parent / "meta_data"
CFG_PATH = Path(__file__).parent / "meta_cfg"

def run_control_tune(benchmark, sysid_trajs, surr_trajs, model, transformer, output_dir, n_iters=100, seed=0):
    # Create controller
    controller = Controller(benchmark.system)
    controller.set_ocp_transformer(transformer)
    controller.set_optimizer(IterativeLQR(benchmark.system))

    # Prepare the model
    model.train(sysid_trajs)
    fixed_model = FixedModel(model, "fixed_model")
    controller.set_model(fixed_model)

    # Create surrogate
    surrogate = MLP(benchmark.system)
    surrogate.train(sysid_trajs)
    fixed_surrogate = FixedModel(surrogate, "mlp_surrogate")

    # Create surrogate
    tuner = ControlTuner(surrogate=fixed_surrogate)
    tuned_controller, tune_result = tuner.run(
        controller=controller,
        tasks=benchmark.task,
        trajs=[],
        n_iters=n_iters,
        rng=np.random.default_rng(seed),
        truedyn=benchmark.dynamics,
        output_dir=output_dir,
        eval_timeout=1200,
    )

    return tune_result

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
    system, trajs = load_data(path=DATA_PATH, name=args.system_name)
    benchmark = GymExtensionBenchmark(name=args.system_name)
    transformer = get_ocp_transformer(system)

    if args.cfg_name:
        cfg = load_cfg(path=CFG_PATH, name=args.cfg_name)
    else:
        cfg = None

    if args.quick:
        trajs = trajs[:10]
        benchmark.task.set_num_steps(20)

    sysid_trajs = trajs[:len(trajs)//2]
    surr_trajs = trajs[len(trajs)//2:]

    model = AutoSelectModel(system)
    if cfg:
        model.set_config(cfg)
    else:
        model.set_config(model.get_config_space().get_default_configuration())

    out_root = Path(args.output_dir)

    n_iters = 100 if not args.quick else 2

    tune_result = run_control_tune(benchmark, sysid_trajs, surr_trajs, model, transformer, out_root / "autompc_data", n_iters)

    with open(out_root / "tune_result.pkl", "wb") as f:
        pickle.dump(tune_result, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--system_name", "-s")
    parser.add_argument("--cfg_name", "-c")
    parser.add_argument("--output_dir", "-o")
    parser.add_argument("--quick", "-q", action="store_true")
    args = parser.parse_args()

    run_experiment(args)
