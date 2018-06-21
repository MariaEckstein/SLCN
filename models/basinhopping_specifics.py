import numpy as np


class AlienBounds(object):
    def __init__(self, xmax, xmin):
        self.xmax = xmax
        self.xmin = xmin

    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x < self.xmax))
        tmin = bool(np.all(x > self.xmin))
        return tmax and tmin


class AlienTakeStep(object):
    def __init__(self, stepsize=0.5, alpha_bounds=np.array([0, 1])):
        self.stepsize = stepsize
        self.alpha_bounds = alpha_bounds

    def __call__(self, x):
        s = self.stepsize
        new_x0 = x[0] + np.random.uniform(-s, s)
        x[0] = max(min(new_x0, self.alpha_bounds[1] - 1e-5), self.alpha_bounds[0] + 1e-5)

        return x
