import os
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from typing import Union,List,Tuple
from collections import defaultdict

from .control_evaluator import ControlEvaluator,StandardEvaluator,ControlEvaluationTrial
from ..task import Task
from .data_store import DataStore
from .parallel_utils import ParallelBackend, JoblibBackend
from ..policy import Policy
from ..dynamics import Dynamics

class ParallelEvaluator(ControlEvaluator):
    """A ControlEvaluator that runs one or more dynamics models and one or more
    tasks and performs them in parallel
    """
    def __init__(self, evaluator : StandardEvaluator,
                dynamics : Union[Dynamics,List[Dynamics]],
                data_store: DataStore,
                backend : ParallelBackend = None):
        if isinstance(evaluator,ParallelEvaluator):
            raise ValueError("Can't pass ParallelEvaluator to ParallelEvaluator constructor")
        super().__init__(evaluator.system, evaluator.tasks)
        self.evaluator = evaluator
        self.data_store = data_store
        if not hasattr(dynamics,'__iter__'):
            dynamics = [dynamics]
        dynamics = [data_store.wrap(dyn) for dyn in dynamics]
        self.dynamics_models = dynamics
        if backend is None:
            self.backend = JoblibBackend(n_jobs=os.cpu_count())
        else:
            self.backend = backend
    
    def cleanup(self):
        for dyn in self.dynamics_models:
            dyn.cleanup()
    
    def num_jobs(self) -> int:
        # return min(self.max_jobs,len(self.dynamics_models)*len(self.tasks))
        return len(self.dynamics_models)*len(self.tasks)

    def run_job(self, controller, job_idx) -> ControlEvaluationTrial:
        model_idx, task_idx = job_idx // len(self.tasks), job_idx % len(self.tasks)
        surrogate = self.dynamics_models[model_idx]
        print("Simulating Surrogate Trajectory for Model {}, Task {}: ".format(model_idx, task_idx))
        self.evaluator.dynamics = surrogate.unwrap()
        if hasattr(controller, "unwrap"):
            controller = controller.unwrap()
        if hasattr(controller,'set_ocp'):  #it's a Controller
            controller.set_ocp(self.tasks[task_idx])
        controller.reset()
        result = self.evaluator.evaluate_for_task(controller, self.tasks[task_idx])
        return self.data_store.wrap(result)

    def __call__(self, controller : Policy):
        print("Entering Parallel Evaluation")
        if hasattr(controller,'model') and hasattr(controller.model,'set_device'):
            controller.model.set_device("cpu")
        for dyn in self.dynamics_models:
            if hasattr(dyn,'set_device'):
                dyn.set_device("cpu")
        wrapped_controller = self.data_store.wrap(controller)
        wrapped_results = self.backend.map(partial(self.run_job, wrapped_controller), range(self.num_jobs()))
        wrapped_controller.cleanup()
        results = [res.unwrap() for res in wrapped_results]
        for res in wrapped_results:
            res.cleanup()
        return results

    def evaluate_for_task(self, controller : Policy, task : Task):
        old_tasks = self.tasks
        self.tasks = [task]
        try:
            res = self(controller)
        except Exception:
            print("Error evaluating for single task")
        finally:
            self.tasks = old_tasks
        return res