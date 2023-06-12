# Standard library includes
import unittest
import sys
from abc import ABC, abstractmethod

# Internal library inlcudes
sys.path.insert(0, "..")
import autompc as ampc
from autompc.sysid import MLP, ARX, SINDy, ApproximateGPModel, Koopman
# DEBUG
from autompc.sysid.mlp import ARMLP
from autompc.benchmarks import DoubleIntegratorBenchmark


# External library includes
import numpy as np

class GenericDiffTest(ABC):
    def setUp(self):
        self.benchmark = DoubleIntegratorBenchmark()
        self.trajs = self.benchmark.gen_trajs(seed=100, n_trajs=20, traj_len=20)
        self.model = self.get_model(self.benchmark.system)

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_configs_to_test(self):
        """Return non-default configs to test, default is
        always included.  Returned values should be dictionary, mapping
        label name to config."""
        raise NotImplementedError

    def get_inputs(self, model, num_inputs=100, seed=100):
        states = []
        controls = []
        ctrl_bounds = self.benchmark.task.get_ocp().get_ctrl_bounds()
        rng = np.random.default_rng(seed)
        for traj in self.trajs:
            for i in range(1, len(traj)):
                states.append(model.traj_to_state(traj[:i]))
                controls.append(rng.uniform(ctrl_bounds[0,0], ctrl_bounds[0,1]))
        states = np.array(states)
        controls = np.array(controls)
        if len(controls.shape) == 1:
            controls = controls.reshape((-1,1))
        selected_idxs = rng.choice(states.shape[0], num_inputs, replace=False)

        input_states = states[selected_idxs, :]
        input_controls = controls[selected_idxs, :]

        # Generate random unit direction vectors to take finite difference in
        dir_states = rng.random(size=input_states.shape)
        dir_states /= np.linalg.norm(dir_states, axis=-1)[:,np.newaxis]
        dir_controls = rng.random(size=input_controls.shape)
        dir_controls /= np.linalg.norm(dir_controls, axis=-1)[:,np.newaxis]
        return input_states, input_controls, dir_states, dir_controls

    def test_model_train_predict(self):
        configs = self.get_configs_to_test()
        configs["default"] = self.model.get_default_config()

        for label, config in configs.items():
            model = self.model.clone()
            model.set_config(config)
            model.train(self.trajs)
            input_states, input_controls, dir_states, dir_controls = self.get_inputs(model)

            # Test Single Prediction
            preds1 = []
            state_jacs1 = []
            ctrl_jacs1 = []
            for state, control in zip(input_states, input_controls):
                pred, state_jac, ctrl_jac = model.pred_diff(state, control)
                preds1.append(pred)
                state_jacs1.append(state_jac)
                ctrl_jacs1.append(ctrl_jac)
            preds1 = np.array(preds1)
            state_jacs1 = np.array(state_jacs1)
            ctrl_jacs1 = np.array(ctrl_jacs1)

            # Test Batch Prediction
            preds2, state_jacs2, ctrl_jacs2 = model.pred_diff_batch(input_states, input_controls)
            self.assertTrue(np.allclose(preds1, preds2))
            self.assertTrue(np.allclose(state_jacs1, state_jacs2))
            self.assertTrue(np.allclose(ctrl_jacs1, ctrl_jacs2))

            # Compare to pre-computed state_jacs and control_jacs
            # Finite Difference            
            eps = 1e-6
            diff_states = model.pred_batch(input_states+eps*dir_states, input_controls) - model.pred_batch(input_states-eps*dir_states, input_controls)
            diff_states /= 2*eps
            diff_controls = model.pred_batch(input_states, input_controls+eps*dir_controls) - model.pred_batch(input_states, input_controls-eps*dir_controls)
            diff_controls /= 2*eps
            self.assertEqual(input_states.shape[1], model.state_dim)


            for i, (dir_state, state_jac1) in enumerate(zip(dir_states, state_jacs1)):
                self.assertTrue(np.allclose((dir_state@state_jac1.T), diff_states[i]))

            for i, (dir_control, control_jac1) in enumerate(zip(dir_controls, ctrl_jacs1)):
                self.assertTrue(np.allclose((dir_control@control_jac1.T), diff_controls[i]))
                
        print('All tests successfully completed')

class MLPTest(GenericDiffTest, unittest.TestCase):
    def get_model(self, system):
        return MLP(system)

    def get_configs_to_test(self):
        relu_config = self.model.get_default_config()
        relu_config["nonlintype"] = "relu"
        tanh_config = self.model.get_default_config()
        tanh_config["nonlintype"] = "tanh"
        sigmoid_config = self.model.get_default_config()
        sigmoid_config["nonlintype"] = "sigmoid"
        return {"relu" : relu_config,
                "tanh" : tanh_config,
                "sigmoid" : sigmoid_config}

class ARMLPTest(GenericDiffTest, unittest.TestCase):
    def get_model(self, system):
        return ARMLP(system)
    
    def get_configs_to_test(self):
        relu_config = self.model.get_default_config()
        relu_config["nonlintype"] = "relu"
        tanh_config = self.model.get_default_config()
        tanh_config["nonlintype"] = "tanh"
        sigmoid_config = self.model.get_default_config()
        sigmoid_config["nonlintype"] = "sigmoid"
        return {"relu" : relu_config,
                "tanh" : tanh_config,
                "sigmoid" : sigmoid_config}
class ARXTest(GenericDiffTest, unittest.TestCase):
    def get_model(self, system):
        return ARX(system)

    def get_configs_to_test(self):
        return dict()

class SINDyTest(GenericDiffTest, unittest.TestCase):
    def get_model(self, system):
        return SINDy(system, allow_cross_terms=True)

    def get_configs_to_test(self):
        poly_config = self.model.get_default_config()
        poly_config["poly_basis"] = "true"
        poly_config["poly_degree"] = 3
        poly_config["poly_cross_terms"] = "true"
        
        trig_config = self.model.get_default_config()
        trig_config["trig_basis"] = "true"
        trig_config["trig_freq"] = 3
        trig_config["trig_interaction"] = "true"

        trig_and_poly_config = self.model.get_default_config()
        trig_and_poly_config["poly_basis"] = "true"
        trig_and_poly_config["poly_degree"] = 2
        trig_and_poly_config["trig_basis"] = "true"
        trig_and_poly_config["trig_freq"] = 2

        return {"poly" : poly_config,
                "trig" : trig_config,
                "trig_and_poly" : trig_and_poly_config
                }

class ApproximateGPTest(GenericDiffTest, unittest.TestCase):
    def get_model(self, system):
        return ApproximateGPModel(system)

    def get_configs_to_test(self):
        return dict()

class KoopmanTest(GenericDiffTest, unittest.TestCase):
    def get_model(self, system):
        return Koopman(system, allow_cross_terms=True)

    def get_configs_to_test(self):
        poly_config = self.model.get_default_config()
        poly_config["poly_basis"] = "true"
        poly_config["poly_degree"] = 3
        poly_config["poly_cross_terms"] = "true"
        
        trig_config = self.model.get_default_config()
        trig_config["trig_basis"] = "true"
        trig_config["trig_freq"] = 3
        trig_config["trig_interaction"] = "true"

        trig_and_poly_config = self.model.get_default_config()
        trig_and_poly_config["poly_basis"] = "true"
        trig_and_poly_config["poly_degree"] = 2
        trig_and_poly_config["trig_basis"] = "true"
        trig_and_poly_config["trig_freq"] = 2

        lasso_config = self.model.get_default_config()
        lasso_config["method"] = "lasso"
        lasso_config["poly_basis"] = "true"
        lasso_config["poly_degree"] = 2
        lasso_config["trig_basis"] = "true"
        lasso_config["trig_freq"] = 2

        stable_config = self.model.get_default_config()
        stable_config["method"] = "stable"
        stable_config["poly_basis"] = "true"
        stable_config["poly_degree"] = 2
        stable_config["trig_basis"] = "true"
        stable_config["trig_freq"] = 2

        return {"poly" : poly_config,
                "trig" : trig_config,
                "trig_and_poly" : trig_and_poly_config,
                "stable" : stable_config,
                "lasso" : lasso_config}