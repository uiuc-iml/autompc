
# Standard library includes
from collections import namedtuple
import pickle
import queue
import multiprocessing
import contextlib
import datetime
import time
import os, sys, glob, shutil
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from functools import partial

# Internal project include
from ..task import Task
from ..sysid.model import Model
from ..sysid.autoselect import AutoSelectModel
from ..controller import Controller
from ..system import System
from ..trajectory import Trajectory
from ..dynamics import Dynamics
from .model_tuner import ModelTuner, ModelTunerResult
from .model_evaluator import ModelEvaluator
from .control_evaluator import ControlEvaluator, StandardEvaluator, ControlEvaluationTrial, trial_to_json
from .control_performance_metric import ControlPerformanceMetric,ConfidenceBoundPerformanceMetric
from .smac_runner import SMACRunner
from .data_store import DataStore, FileDataStore, WrappedData
from .parallel_utils import ParallelBackend,SerialBackend
from .parallel_evaluator import ParallelEvaluator

# External library includes
import numpy as np
import pynisher
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


ControlTunerResult = namedtuple("ControlTunerResult", ["inc_cfg", "cfgs", 
    "inc_cfgs", "costs", "inc_costs", "truedyn_costs", "inc_truedyn_costs", 
    "truedyn_infos", "surr_infos", 
    "surr_tune_result"]) # TODO update docstring
"""
PipelineTuneResult contains information about a tuning process.

.. py:attribute:: inc_cfg
    
    The final tuned configuration.

.. py:attribute:: cfgs

    List of configurations. The configuration evaluated at each
    tuning iteration.

.. py:attribute:: inc_cfgs

    The incumbent (best found so far) configuration at each
    tuning iteration.

.. py:attribute:: costs

    The cost observed at each iteration of
    tuning iteration.

.. py:attribute:: inc_costs

    The incumbent score at each tuning iteration.

.. py:attribute:: truedyn_costs

    The true dynamics cost observed at each iteration of
    tuning iteration. None if true dynamics was not provided.

.. py:attribute:: inc_truedyn_costs

    The incumbent true dynamics cost observed at each iteration of
    tuning iteration. None if true dynamics was not provided.

.. py:attribute:: surr_trajs

    The trajectory simulated at each tuning iteration with respect to the
    surrogate model.

.. py:attribute:: truedyn_trajs

    The trajectory simulated at each tuning iteration with respect to the
    true dynamics. None if true dynamics are not provided.

.. py:attribute:: surr_tune_result

    The ModelTuneResult from tuning the surrogate model, for modes "autotune"
    and "autoselect".  None for other tuning modes.

"""

#autoselect_factories = [MLPFactory, SINDyFactory, ApproximateGPModelFactory,
#        ARXFactory, KoopmanFactory]

def get_tune_result(surr_tune_history, control_tune_history):
    cfgs, inc_cfgs, costs, inc_costs, truedyn_costs, inc_truedyn_costs, \
        surr_infos, truedyn_infos =  [], [], [], [], [], [], [], []
    inc_cost = float("inf")
    inc_cfg = None

    for key, val in control_tune_history.data.items():
        #parse additional data from additional_info dict
        cfg = control_tune_history.ids_config[key.config_id]
        if not val.additional_info:
            continue
        if val.cost < inc_cost:
            inc_cost = val.cost
            if "truedyn_cost" in val.additional_info:
                inc_truedyn_cost = val.additional_info["truedyn_cost"]
            inc_cfg = cfg
        inc_costs.append(inc_cost)
        inc_cfgs.append(inc_cfg)
        cfgs.append(cfg)
        costs.append(val.cost)
        surr_infos.append(val.additional_info["surr_info"])
        if "truedyn_cost" in val.additional_info:
            inc_truedyn_costs.append(inc_truedyn_cost)
            truedyn_costs.append(val.additional_info["truedyn_cost"])
            truedyn_infos.append(val.additional_info["truedyn_info"])

    if surr_tune_history:
        surr_tune_result = surr_tune_history[-1]
    else:
        surr_tune_result = None

    tune_result = ControlTunerResult(inc_cfg = inc_cfg,
            cfgs = cfgs,
            inc_cfgs = inc_cfgs,
            costs = costs,
            inc_costs = inc_costs,
            truedyn_costs = truedyn_costs,
            inc_truedyn_costs = inc_truedyn_costs,
            surr_infos = surr_infos,
            truedyn_infos = truedyn_infos,
            surr_tune_result = surr_tune_result)

    return tune_result

class ControlTuner:
    """
    This class tunes controllers, searching over the hyperparameters for the
    model, optimizer, and ocp_factory in order to optimize a performance metric.

    There are three major tuning modes:

    - End-to-end: tunes end-to-end performance of model, cost, and optimizer.
      Specify tuning_mode='default' or 'endtoend'.
    - Two-stage: tunes the model for accuracy first, then tunes the cost
      and optimizer assuming the model is fixed.  This is usually faster
      because the hyperparameter space for each stage is smaller.
      Specify tuning_mode='twostage'.
    - Fixed-model: uses a given model and tunes the cost and optimizer only.
      Use this only if you know the dynamics model.  Specify surrogate=X where
      X is a Dynamics instance or a trained Model instance.

    In all but fixed-model mode, cross-simulation is performed.  In cross-
    simulation, one or more surrogate models are learned on a holdout set,
    and the controller's models are learned from the remainder.  The surrogate
    model is auto-tuned on the holdout set.  The controller's
    performance is tuned on the surrogate model(s).

    The surrogate model is tuned on the holdout set according to some accuracy-
    related performance metric.  By default, a CrossValidationModelEvaluator with
    surrogate_tune_folds folds is used. 
    
    - To use holdout evaluation, you can use surrogate_tune_folds = 1 and specify
      surrogate_tune_holdout.
    - To use a custom evaluator, you can specify surrogate_evaluator=X. 
    - If you want to fix the surrogate model class, pass in surrogate=X where X
      is a Model instance (untrained). 
    - If you want to fix the class and its hyperparameters, pass in surrogate=X
      and call `freeze_hyperparameters()` on X.
    
    In other words, the arguments `surrogate_` map directly to the arguments of
    ModelTuner.

    The cross-simulation method by default runs a StandardEvaluator.  If
    control_tune_boostraps > 1, multiple surrogates are tuned and a ParallelEvaluator
    evaluates the cartesian product of all tasks and .
    To completely override this behavior, specify control_evaluator=X. 
    
    The performance metric is encapsulated by a ControlPerformanceMetric subclass. 
    The default tuner will minimize the performance distribution across surrogates
    and tasks at quantile `evaluation_quantile`.

    To freeze some aspects of the Controller, pass in controller=X where X is a
    configured Controller with the desired `set_optimizer_hyper_values` and
    `set_ocp_transformer_hyper_values` items fixed.
    """
    def __init__(self, tuning_mode="default", surrogate : Optional[Model] = None,
            surrogate_split=0.5,
            surrogate_tune_holdout=0.25, surrogate_tune_folds=3, surrogate_tune_metric="rmse", surrogate_tune_horizon=1, surrogate_tune_quantile=None,
            surrogate_evaluator : Optional[ModelEvaluator]=None,
            control_tune_bootstraps=1, max_trials_per_evaluation=None, control_evaluator : Optional[ControlEvaluator]=None,
            performance_quantile=0.5, performance_eval_time_weight=0.0, performance_infeasible_cost=100.0,
            performance_metric : Optional[ControlPerformanceMetric]=None, parallel_backend : ParallelBackend = None): 
        """
        Parameters
        ----------
        tuning_mode : string
            Mode for selecting tuning mode
            "default" or "endtoend" - whole controller will be tuned at once
            "twostage" - controller model will be tuned first, then the cost
                and optimizer
        surrogate : Model
            Surrogate model. If None, a model will be auto-selected in the first step.
            If tunable, this will be tuned in the first step.  If learnable
            and has not been learned, then this will be passed to a holdout or bootstrap
            control evaluator.
        surrogate_split : float
            The fraction of examples used for training the surrogate vs
            the controller's sysID model. If no surrogate tuning is
            performed, then all the data will be used for control tuning.
        surrogate_tune_holdout : float
            Passed into ModelTuner as the eval_holdout argument.
        surrogate_tune_folds : int
            Passed into ModelTuner as the eval_folds argument.
        surrogate_tune_metric : int
            Passed into ModelTuner as the eval_metric argument.
        surrogate_tune_horizon : int
            Passed into ModelTuner as the horizon argument.
        surrogate_tune_quantile : float or None
            Passed into ModelTuner as the eval_quantile argument.
        surrogate_evaluator : int
            Passed into ModelTuner as the evaluator argument.  If given,
            ignores the prior settings.
        control_tune_bootstraps : int
            If > 1 and control_evaluator = None, will use a
            BootstrapControlEvaluator. Otherwise will use a
            StandardControlEvaluator.
        max_trials_per_evaluation : int
            If given and control_evaluator = None, at most this # of tasks and 
            surrogates will be evaluated.
        control_evaluator : ControlEvaluator
            Overrides the method by which controllers will be evaluated and ignores
            the prior settings.
        performance_quantile : float
            Optimizes this quantile of the distribution of task / surrogate
            performance.
        performance_eval_time_weight : float
            Penalizes the evaluation time of the controller (in seconds) by
            this weight in the overall performance metric.
        performance_infeasible_cost : float
            Penalizes infeasible states / controls by this much in the overall
            performance metric
        performance_metric : ControlPerformanceMetric
            Overrides the default ConfidenceBoundPerformanceMetric and ignores the
            prior settings.
        parallel_backend : ParallelBackend
            Uses a specific parallel backend for parallelization.  If None, uses
            SerialBackend.
        """
        self.tuning_mode = tuning_mode
        self.surrogate = surrogate
        self.surrogate_tune_holdout = surrogate_tune_holdout
        self.surrogate_tune_folds = surrogate_tune_folds
        self.surrogate_tune_metric = surrogate_tune_metric
        self.surrogate_tune_horizon = surrogate_tune_horizon
        self.surrogate_tune_quantile = surrogate_tune_quantile
        self.surrogate_evaluator = surrogate_evaluator
        self.surrogate_split = surrogate_split
        self.control_tune_bootstraps = control_tune_bootstraps
        self.max_trials_per_evaluation = max_trials_per_evaluation
        self.control_evaluator = control_evaluator
        if performance_metric is None:
            performance_metric = ConfidenceBoundPerformanceMetric(quantile=performance_quantile,eval_time_weight=performance_eval_time_weight,infeasible_cost=performance_infeasible_cost)
            # performance_metric = ControlPerformanceMetric()
        self.performance_metric = performance_metric
        self.parallel_backend = parallel_backend
        if self.parallel_backend is None:
            self.parallel_backend = SerialBackend()

    def _prepare_surrogates(self, system : System, surr_trajs : List[Trajectory],
                            rng, surrogate_tune_iters : int,
                            data_store: DataStore) -> Tuple[List[WrappedData],Optional[ModelTunerResult]]:
        #Surrogate options
        # 1. Surrogate provided as fixed dynamics model or trained model.
        #    Bootstrap evaluation not enabled.
        # 2. Surrogate provided as fixed (tuned?) model class but untrained model.
        #    In this case, we need to train the surrogate OR use bootstrap evaluator
        #    to train multiple surrogates.
        # 3. Surrogate provided as un-tuned model or None (latter case we auto-select).
        #    In this case, we need to tune the surrogate.  We can train the surrogate
        #    OR use bootstrap evaluator to train multiple surrogates.
        #
        # Result:
        #    One or more trained, wrapped Models used for surrogates
        surrogate = self.surrogate
        if surrogate is None:
            surrogate = AutoSelectModel(system)
        trained_wrapped_surrogates = []
        tune_result = None
        
        if (isinstance(surrogate, Dynamics) and not isinstance(surrogate, Model)) or surrogate.is_trained:
            print("------------------------------------------------------------------")
            print("Skipping surrogate tuning, surrogate is a trained model")
            print("------------------------------------------------------------------")
            trained_wrapped_surrogates = [data_store.wrap(surrogate)]
        else:
            if surrogate.is_tunable():
                # Run surrogate tuning
                print("------------------------------------------------------------------")
                print("Beginning surrogate tuning with model class",surrogate.name)
                print("------------------------------------------------------------------")
                tuner = ModelTuner(system, surr_trajs, surrogate, eval_holdout=self.surrogate_tune_holdout, eval_folds=self.surrogate_tune_folds,
                    eval_metric=self.surrogate_tune_metric,eval_horizon=self.surrogate_tune_horizon,eval_quantile=self.surrogate_tune_quantile,evaluator=self.surrogate_evaluator)
                model, tune_result = tuner.run(rng, surrogate_tune_iters, retrain_full=False)
                print("------------------------------------------------------------------")
                print("Auto-tuned surrogate model class:")
                print(tune_result.inc_cfg)
                print("Cost",tune_result.inc_costs[-1])
                print("------------------------------------------------------------------")
                surrogate = model
            else:
                # No need for surrogate tuning, but would need training
                pass
        
        if self.control_tune_bootstraps > 1:
            def _train_bootstrap(bootstrap_sample: WrappedData, base_surrogate: WrappedData, data_store: DataStore):
                bootstrap_sample = bootstrap_sample.unwrap()
                base_surrogate = base_surrogate.unwrap()
                bootstrap_surrogate = base_surrogate.clone()
                bootstrap_surrogate.train(bootstrap_sample)
                bootstrap_surrogate = data_store.wrap(bootstrap_surrogate)
                return bootstrap_surrogate

            #perform bootstrap sample to create surrogate ensemble
            population = np.empty(len(surr_trajs), dtype=object)
            for i, traj in enumerate(surr_trajs):
                population[i] = surr_trajs
            bootstrap_samples = []
            for i in range(self.control_tune_bootstraps):
                bootstrap_samples.append(data_store.wrap(rng.choice(population, len(surr_trajs), replace=True, axis=0)))
            surrogate_dynamics = self.parallel_backend.map(partial(_train_bootstrap, base_surrogate=data_store.wrap(surrogate), data_store=data_store), bootstrap_samples)
            trained_wrapped_surrogates = surrogate_dynamics #already wrapped!
        else:
            surrogate.train(surr_trajs)
            trained_wrapped_surrogates = [data_store.wrap(surrogate)]

        return trained_wrapped_surrogates,tune_result

        
    def _get_cfg_evaluator(self, controller : Controller, tasks : List[Task], trajs : List[Trajectory],
                            truedyn : Optional[Dynamics], rng, surrogate_tune_iters : int, data_store: DataStore):
        # After setting up surrogate, can either have one or more tuned surrogate models.
        # May also have true dynamics (used just for benchmarking, not available in real
        # systems).
        #
        # Controller tuning options
        # 1. Two-stage tuning: tune the controller's model first, then tune the cost and optimizer
        # 2. End-to-end tuning: tune the controller's model, cost, and optimizer all at once 
        #
        # The config evaluator will accept:
        # 1. The one or more surrogates
        # 2. The optional true dynamics
        # 3. Either A) a controller with a fixed model, or B) a controller with a free model + tuning trajectories
        # 4. Information about parallel processing context.
        #
        # It needs to be able to be run in a Multiprocessing instance.
        control_evaluator = self.control_evaluator
        surrogate_split = self.surrogate_split
        surrogate = self.surrogate
        if surrogate is not None and ((isinstance(surrogate, Dynamics) and not isinstance(surrogate, Model)) or surrogate.is_trained):
            # No need for surrogate training or tuning
            surrogate_split = 0.0
        
        surr_size = int(surrogate_split * len(trajs))
        shuffled_trajs = trajs[:]
        rng.shuffle(shuffled_trajs)
        surr_trajs = shuffled_trajs[:surr_size]
        sysid_trajs = shuffled_trajs[surr_size:]

        wrapped_surrogates,train_surr_info = self._prepare_surrogates(controller.system,surr_trajs,rng,surrogate_tune_iters,data_store)
        
        if truedyn is not None:
            wrapped_truedyn = data_store.wrap(truedyn)
        else:
            wrapped_truedyn = None

        if self.tuning_mode == 'twostage':
            #tune the model if not already tuned by surrogate tuning
            print("------------------------------------------------------------------")
            print("Beginning two-stage tuning")
            print("------------------------------------------------------------------")
            if surrogate is None or isinstance(self.surrogate,AutoSelectModel):
                print("Reusing surrogate model tuning for SysID model to save time")
                model = wrapped_surrogates[0].unwrap()
            else:
                surrogate = wrapped_surrogates[0].unwrap()
                print("Tuning MPC model")
                tuner = ModelTuner(controller.system, trajs, surrogate, eval_holdout=self.surrogate_tune_holdout, eval_folds=self.surrogate_tune_folds,
                    eval_metric=self.surrogate_tune_metric,eval_horizon=self.surrogate_tune_horizon,eval_quantile=self.surrogate_tune_quantile,evaluator=self.surrogate_evaluator)
                model, tune_result = tuner.run(rng, surrogate_tune_iters, retrain_full=False)
                print("------------------------------------------------------------------")
                print("Auto-tuned SysID model class:")
                print(tune_result.inc_cfg)
                print("Cost",tune_result.inc_costs[-1])
                print("------------------------------------------------------------------")
            controller.fix_option('model',model.name)
            controller.set_model_hyper_values(model.name,**model.get_config())
            print()

        if control_evaluator is not None:
            control_evaluator.tasks = tasks
            control_evaluator.dynamics = wrapped_surrogates
        
        cfg_evaluator = ControlCfgEvaluator(
            controller=data_store.wrap(controller),
            tasks=tasks,
            performance_metric=self.performance_metric,
            sysid_trajs=data_store.wrap(sysid_trajs),
            control_evaluator=control_evaluator,
            surrogate_models=wrapped_surrogates,
            truedyn=wrapped_truedyn,
            paralle_backend=self.parallel_backend,
            data_store=data_store,
            train_surr_info=train_surr_info
        )

        return cfg_evaluator


    def run(self, controller, tasks, trajs, n_iters, rng, truedyn=None, 
            surrogate_tune_iters=100, eval_timeout=600, output_dir=None, restore_dir=None,
            save_all_controllers=False, use_default_initial_design=True):
        """
        Run tuning.

        Parameters
        ----------
        controller : Controller
            Controller to tune.  Can be an AutoSelectController or a Controller
            with a manually-configured configuration space.

        tasks : Task or [Task]
            Tasks which specify the tuning problem.

        trajs : List of Trajectory
            Trajectory training set.

        n_iters : int
            Number of tuning iterations

        rng : numpy.random.Generator
            RNG to use for tuning.

        truedyn : obs, ctrl -> obs
            True dynamics function. If passed, the true dynamics cost
            will be evaluated for each iteration in addition to the surrogate
            cost. However, this information will not be used for tuning.

        surrogate_tune_iters : int
            Number of iterations to use for surrogate tuning. Used for "autotune"
            and "autoselect" modes. Default is 100

        eval_timeout : float
            Maximum number of seconds to allow for tuning of a single configuration.
            Default is 600.

        output_dir : str
            Output directory to store intermediate tuning information.  If None,
            a filename will be automatically selected.

        restore_dir : str
            To resume a previous tuning process, pass the output_dir for the previous
            tuning process here. 

        Returns
        -------
        controller : Controller
            Final tuned controller.  (If deploying, you should run controller.build(trajs) to
            re-train on the whole dataset)

        tune_result : PipelineTuneResult
            Additional tuning information.
        """
        # TODO handle save_all_controllers
        smac_runner = SMACRunner(output_dir, restore_dir, use_default_initial_design)
        data_store = FileDataStore(smac_runner.data_dir)
        
        if isinstance(tasks, Task):
            tasks = [tasks]

        if not smac_runner.restore:
            controller = controller.clone()
            controller.set_ocp(tasks[0].get_ocp())
            cfg_evaluator = self._get_cfg_evaluator(controller, tasks, trajs, truedyn, rng, surrogate_tune_iters, data_store)
        else:
            cfg_evaluator = smac_runner.restore_cfg_evaluator()

        cs = controller.get_config_space()

        inc_cfg, run_history  = smac_runner.run(cs, cfg_evaluator, n_iters, rng, eval_timeout)

        # Generate final model and controller
        tuned_controller = controller.clone()
        tuned_controller.set_ocp(tasks[0].get_ocp())
        tuned_controller.set_config(inc_cfg)
        trajs = cfg_evaluator.sysid_trajs.unwrap()
        tuned_controller.build(trajs)

        tune_result = get_tune_result(cfg_evaluator.train_surr_info, run_history)

        return tuned_controller, tune_result

@dataclass
class ControlCfgEvaluator:
    controller: WrappedData     #Controller
    tasks : List[Task]
    performance_metric: ControlPerformanceMetric
    sysid_trajs: WrappedData    #List[Trajectory]
    control_evaluator: ControlEvaluator
    surrogates : WrappedData    #List[Model]
    truedyn: WrappedData        #Dynamics
    parallel_backend : ParallelBackend
    data_store: DataStore
    train_surr_info: Optional[ModelTunerResult] = None

    def __call__(self, cfg):
        print("\n>>> ", datetime.datetime.now(), "> Evaluating Cfg: \n", cfg)
        
        #unpack everything
        controller = controller.unwrap()
        trajs = self.sysid_trajs.unwrap()
        if self.control_evaluator is None:
            if len(self.surrogates)*len(self.tasks) == 1:
                self.control_evaluator = StandardEvaluator(controller.system, self.tasks, self.surrogates[0].unwrap())
            else:
                self.control_evaluator = ParallelEvaluator(StandardEvaluator(controller.system, self.tasks), self.surrogates, self.data_store, self.parallel_backend)
        truedyn_evaluator = None
        if self.truedyn is not None:
            truedyn_std_evaluator = StandardEvaluator(controller.system, self.tasks, self.truedyn.unwrap())
            if len(self.tasks)==1:
                truedyn_evaluator = truedyn_std_evaluator
            else:
                truedyn_evaluator = ParallelEvaluator(truedyn_std_evaluator, self.truedyn, self.data_store, self.parallel_backend)
        
        #test controller
        info = dict()
        controller = self.controller.clone()
        controller.set_config(cfg)
        controller.build(trajs)
        print("Run Controller Evaluation...")
        trials = self.control_evaluator(controller)
        #evaluate trials and extract run performance
        performance = self.performance_metric(trials)
        info["surr_cost"] = performance
        info["surr_info"] = list(map(trial_to_json, trials))
        if truedyn_evaluator is not None:
            truedyn_trials = truedyn_evaluator(controller)
            performance = self.performance_metric(truedyn_trials)
            info["truedyn_cost"] = performance
            info["truedyn_info"] = list(map(trial_to_json, truedyn_trials))
    
        # if self.controller_save_dir:
        #     controller_save_fn = os.path.join(self.controller_save_dir, "controller_{}.pkl".format(self.eval_number))
        #     with open(controller_save_fn, "wb") as f:
        #         pickle.dump(controller, f)

        return info["surr_cost"], info