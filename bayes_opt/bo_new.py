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
            raise ValueError("Cannot retrieve next object from empty queue.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class BayesianOptimization:
    def __init__(self, f, pbounds, random_state=None, verbose=1):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self._random_state
        )

        # Event initialization
        events = [Events.INIT_DONE, Events.FIT_STEP_DONE, Events.FIT_DONE]
        super(BayesianOptimization, self).__init__(events)


    def observe(self, x, target):
        """Expect observation with known target"""
        self._space.observe(x, target)

    def inspect(self, x):
        """Probe target of x"""
        self._space.inspect(x)

    def suggest(self, utility_function):
        """Moxt promissing point to probe next"""
        self._gp.fit(self._space.x, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return suggestion

    def _random_suggestion(self):
        """Randomly pick a point to probe in the parameter space."""
        return self._space.random_sample()

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def maximize(self,
                 init_points: int=5,
                 n_iter: int=25,
                 acq: str='ucb',
                 kappa: float=2.576,
                 xi: float=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_queue(init_points)

        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty and iteration < n_iter:

            try:
                x_probe = next(self._queue)
            except ValueError:
                x_probe = self.suggest(util)

            self.inspect(x_probe)
            iteration += 1

