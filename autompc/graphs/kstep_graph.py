# Created by William Edwards

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ..sysid.metrics import get_model_rmse, get_model_rmsmens, get_model_abs_error

class KstepPredAccGraph:
    def __init__(self, system, trajs, kmax, logscale=False, metric="rmse", dimensions="combined"):
        """
        Create k-step model prediction accuracy graph.

        Parameters
        ----------
        system : System
            System on which models are being evaluted

        trajs : List of Trajectory
            Evaluation trajectory set

        kmax : int
            Maximum horizon to evaluate

        logscale : bool
            Use log scale on y-axis if true

        metric : string
            Prediction accuracy metric to use. One of "rmse", "rmsmens", or "abserror"

        dimensions : string, int, or list of ints
            Dimensions to evaluate. One of "combined", an integer index, or a list of
            integer indices. 
        """
        self.kmax = kmax
        self.trajs = trajs
        self.logscale = logscale
        self.models = []
        self.labels = []
        self.plot_kwargs = []
        self.dimensions = dimensions
        if isinstance(dimensions, int):
            self.dimensions = [dimensions]

        if metric == "rmse":
            self.metric = get_model_rmse
        elif metric == "rmsmens":
            self.metric = get_model_rmsmens
        elif metric == "abserror":
            self.metric = get_model_abs_error
        else:
            raise ValueError("Unknown error metric")

    def add_model(self, model, label, plot_kwargs=dict()):
        """
        Add a model for comparison

        Parameters
        ----------
        model : Model
            Model to compare

        label : string
            Label for model
        
        plot_kwargs : dict
            kwargs to pass to matplotlib plot
        """
        self.models.append(model)
        self.labels.append(label)
        self.plot_kwargs.append(plot_kwargs)

            
    def __call__(self, fig, ax):
        """
        Create graph.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure in which to create graph

        ax : matplotlib.axes.Axes
            Axes in which to create graph
        """
        for model, label, kwargs in zip(self.models, self.labels, self.plot_kwargs):    
            if self.dimensions == "combined":
                rmses = [self.metric(model, self.trajs, horizon) 
                            for horizon in range(1, self.kmax)] 
                ax.plot(list(range(1, self.kmax)), rmses, label=label, **kwargs)
            else:
                for d in self.dimensions:
                    rmses = [self.metric(model, self.trajs, horizon, dimension=d) 
                                for horizon in range(1, self.kmax)] 
                    if len(self.dimensions) > 1:
                        label = "%s dim %d"%(label,d)
                    ax.plot(list(range(1, self.kmax)), rmses, label=label, **kwargs)

        ax.set_xlabel("Prediction Horizon")
        if len(self.dimensions) == 1:
            ax.set_ylabel("Prediction Error, dim %d"%self.dimensions[0])
        else:
            ax.set_ylabel("Prediction Error")
        if self.logscale:
            ax.set_yscale("log")

        ax.legend()
