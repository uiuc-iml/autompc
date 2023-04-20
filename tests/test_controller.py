# Created by William Edwards (wre2@illinois.edu), 2023-04-19

# Standard library includes
import unittest
import sys
from abc import ABC, abstractmethod

# External library includes
import numpy as np

# Internal library includes
sys.path.insert(0, "..")
import autompc as ampc
from autompc import Controller
from autompc.benchmarks import DoubleIntegratorBenchmark
from autompc.sysid import MLP
from autompc.optim import IterativeLQR
from autompc.ocp import QuadCostTransformer

class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.benchmark = DoubleIntegratorBenchmark()
        self.trajs = self.benchmark.gen_trajs(seed=100, n_trajs=20, traj_len=20)

    def test_build_and_run(self):
        # Set-up Controller
        controller = Controller(self.benchmark.system)
        controller.set_model(MLP(self.benchmark.system))
        controller.set_optimizer(IterativeLQR(self.benchmark.system))
        controller.set_ocp_transformer(QuadCostTransformer(self.benchmark.system))

        # Test controller build
        controller.set_ocp(self.benchmark.task)
        controller.build(self.trajs)

        self.assertTrue(controller.is_built())
        self.assertTrue(controller.model.is_trained)

        # Test controller step
        control_1 = controller.step(obs=np.zeros(self.benchmark.system.obs_dim))
        control_2 = controller.step(obs=np.ones(self.benchmark.system.obs_dim))

        self.assertEquals(control_1.shape, (self.benchmark.system.ctrl_dim,))
        self.assertEquals(control_2.shape, (self.benchmark.system.ctrl_dim,))

        # Test controller reset
        controller.reset()
        control_1b = controller.step(obs=np.zeros(self.benchmark.system.obs_dim))
        control_2b = controller.step(obs=np.ones(self.benchmark.system.obs_dim))

        self.assertEquals(control_1, control_1b)
        self.assertEquals(control_2, control_2b)