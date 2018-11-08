from __future__ import print_function, division
import numpy as np
from .helpers import ensure_rng, unique_rows


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.observe_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds: dict, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(list(pbounds.values()), dtype=np.float)

        # preallocated memory for X and Y points
        self._x = None
        self._target = None

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._x) == len(self._y)
        return len(self._y)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def x(self):
        return self._x

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def observe(self, x, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = np.asarray(x).reshape(1, -1)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x)] = y

        self._x = np.concatenate([self._x, x])
        self._y = np.concatenate([self._y, y])

    def inspect(self, x):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = np.asarray(x).ravel()
        assert x.size == self.dim, 'x must have the same dimensions'

        try:
            y = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            y = self.target_func(**params)
            self.observe(x, y)
        return y

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def max(self):
        """
        Return the current parameters that best maximize target function with
        that maximum value.
        """
        return {'target': self.target.max(),
                'x': dict(zip(self.keys, self.x[self.target.argmax()]))}

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self._keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]

    def _assert_internal_invariants(self):
        """
        Run internal consistency checks to ensure that data structure
        assumptions have not been violated.
        """
        if self.x is None:
            assert self.target is None
        else:
            assert len(self.x) == len(self.target)
            assert np.all(unique_rows(self.x))
