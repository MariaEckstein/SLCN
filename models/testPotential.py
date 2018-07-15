import numpy as np
import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as T


# Simulate choice data
n_subj = 20
n_trials = 30
true_p = np.arange(0.5, 0.99, (0.99 - 0.5) / n_subj) * np.ones((n_trials, n_subj))
choices = np.random.binomial(n=1, p=true_p, size=(n_trials, n_subj))

# Create model
print('Compiling model...\n')
with pm.Model() as model:

    # Population-wide mean and sd
    mu = pm.Uniform('mu', lower=0, upper=1)
    sigma = pm.HalfNormal('sigma', sd=0.2)

    # Individual deviation from population parameters
    indiv_tilde = pm.Normal('indiv_tilde', mu=0, sd=0.2, shape=n_subj)

    # Individual parameters
    indiv_p = pm.Deterministic('indiv_p', mu + sigma * indiv_tilde)

    # Constraints on individual parameters
    indiv_min = pm.Potential('indiv_min', T.switch(indiv_p < 0, -np.inf, 0))
    indiv_max = pm.Potential('indiv_max', T.switch(indiv_p > 1, -np.inf, 0))

    # Observations to be fitted
    obs = pm.Bernoulli('obs', p=indiv_p * T.ones(shape=(n_trials, n_subj)), observed=choices)

    # Sample trace / fit model
    trace = pm.sample(draws=5000, tune=200, chains=2, cores=1, nuts_kwargs=dict(target_accept=.99))

# Plot results
print('Preparing trace plot...\n')
pm.traceplot(trace)
plt.show()
