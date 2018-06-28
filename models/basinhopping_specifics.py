import numpy as np


class MyBounds(object):
    def __init__(self, xmax, xmin):
        self.xmax = xmax
        self.xmin = xmin

    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x < self.xmax))
        tmin = bool(np.all(x > self.xmin))
        return tmax and tmin


class MyTakeStep(object):
    def __init__(self, stepsize=0.5, bounds=np.array([0, 1])):
        self.stepsize = stepsize
        self.bounds = bounds

    def __call__(self, x):
        s = self.stepsize
        new_x = x + np.random.uniform(-s, s, x.shape)
        new_x[np.array(new_x) < self.bounds[0]] = self.bounds[0] + 1e-3
        new_x[np.array(new_x) > self.bounds[1]] = self.bounds[1] - 1e-3
        return new_x
