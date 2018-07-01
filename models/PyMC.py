import numpy as np
import theano.tensor as tt
import pymc3 as pm
import timeit

import seaborn as sns
import matplotlib.pyplot as plt


data = np.random.randn(100)
with pm.Model() as model:

    # Part 1: Specify model

    # Normal distribution
    mu = pm.Normal('mu', mu=0, sd=1)
    sd = pm.HalfNormal('sd', sd=1)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=data)
    # help(pm.Normal)

    # # Bounded variables (will use LogOdd by default to make sampling more efficient; turn off with transform=None)
    # alpha = pm.Uniform('alpha', lower=0, upper=1)
    #
    # # Tell PyMC to keep track of obs + 2 (can also just use obs + 2 later, but this will keep track of it throughout)
    # obs_plus_two = pm.Deterministic('obs plus 2', obs + 2)
    #
    # # Check random variables (RVs) of the model
    # print(model.basic_RVs)  # [mu, alpha_interval__, obs]; alpha_interval__ is alpha transformed by LogOdd ]-inf, inf[
    # print(model.free_RVs)  # [mu, alpha_interval__]
    # print(model.observed_RVs)  # [obs]
    # print(model.deterministics)  # [alpha, obs plus 2]; alpha is our alpha [0, 1]
    # print(model.logp({'mu': 1, 'alpha_interval__': .5}))
    # print(mu.logp({'mu': 1, 'alpha_interval__': .5}))

    # Available distributions:
    # print(dir(pm.distributions.continuous))
    print(dir(pm.distributions.discrete))  # also has timeseries and mixture

    # Part 2: Inference

    # Note: ALWAYS set cores=1 on my laptop! Will crash will CVS pickle issue otherwise

    # Collect traces
    trace = pm.sample(1000, tune=500, cores=1)  # chains=1,
    # print(len(trace))  # 1000
    # print(trace.nchains)  # 2
    # print(trace.get_values('mu', chains=1).shape)  # (1000,)

    # Available samplers (NUTS should be preferred for almost all continuous models)
    # print(list(filter(lambda x: x[0].isupper(), dir(pm.step_methods))))  # NotablY, Metropolis and Slice

    # # Pass other sampling method to sample
    # step = pm.Metropolis()
    # trace = pm.sample(1000, step=step, cores=1)

    pass

# Part 3: Analyze sampling results

# Plot distributions
pm.traceplot(trace)
plt.show()

# Calculate R-hat
print(pm.gelman_rubin(trace))

# Forest plot
pm.forestplot(trace)
plt.show()

# Plot posterior
pm.plot_posterior(trace)
plt.show()

# # Energy plot
# pm.energyplot(trace)
# plt.show()

# Part 4: Posterior predictive sampling (perform prediction on hold-out data)
with model:
    post_pred = pm.sample_ppc(trace, samples=500, size=len(data))

post_pred['obs'].shape  # sample_ppc() returns a dict with a key for every observed node

plt.figure()
ax = sns.distplot(post_pred['obs'].mean(axis=1), label='Posterior predictive means')
ax.axvline(data.mean(), color='r', ls='--', label='True mean')
ax.legend()
plt.show()