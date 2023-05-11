sysid package
=============

SysID Base Classes 
------------------
 
The Model Class
^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.model.Model
   :members:


Supported System ID Models
--------------------------

Multi-layer Perceptron
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.MLP

Autoregressive Multi-layer Perceptron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.ARMLP

Sparse Identification of Nonlinear Dynamics (SINDy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.SINDy

Autoregression (ARX)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.ARX

Koopman
^^^^^^^

.. autoclass:: autompc.sysid.Koopman

Approximate Gaussian Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.ApproximateGPModel


Model Metrics
^^^^^^^^^^^^^

.. autofunction:: autompc.sysid.metrics.get_model_rmse

.. autofunction:: autompc.sysid.metrics.get_model_rmsmens