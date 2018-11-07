import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .helpers import UtilityFunction, PrintLog, acq_max, ensure_rng
from .target_space import TargetSpace
from .observer import Observable


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise RuntimeError("Cannot retrieve next object from empty queue.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observations:
    def __init__(self):
        self._x = []
        self._target = []

    def add(self, x, target):
        self._x.append(x)
        self._target = target


class BayesianOptimization:
    def __init__(self, f, bounds, random_state=None, verbose=1):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # observations
        self._obs =

        # Counter of iterations
        self._i = 0

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self._random_state
        )

        # non-public config for maximizing the aquisition function
        # (used to speedup tests, but generally leave these as is)
        self._acqkw = {'n_warmup': 100000, 'n_iter': 250}

        # Event initialization
        events = [Events.INIT_DONE, Events.FIT_STEP_DONE, Events.FIT_DONE]
        super(BayesianOptimization, self).__init__(events)


    def observe(self, x, target):
        """Expect observation with known target"""
        pass

    def inspect(self, x):
        """Probe target of x"""
        pass

    def suggest(self):
        """Moxt promissing point to probe next"""
        pass

    def _random_suggestion(self):
        """Randomly pick a point to probe in the parameter space."""
        pass

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

