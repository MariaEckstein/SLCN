import pickle
import datetime
import os

import pymc3 as pm
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from shared_modeling_simulation import get_paths, p_from_Q, update_Q, get_likelihoods, post_from_lik
from modeling_helpers import load_data, get_population_level_priors, get_slopes


# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'test'

# Which model should be run?
fit_age_slopes = False

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 20
n_tune = 1
n_chains = 2
if run_on_cluster:
    n_cores = 8
else:
    n_cores = 1

# Load to-be-fitted data
n_subj, rewards, choices, ages = load_data(run_on_cluster, fitted_data_name, kids_and_teens_only, adults_only, verbose)

# Prepare things for saving
save_dir = get_paths(run_on_cluster)['fitting results']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

now = datetime.datetime.now()
save_id = '_'.join([file_name_suff,
                    str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute),
                    fitted_data_name, 'n_samples' + str(n_samples)])

# Fit model
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

for model_name in ('RL', 'Bayes'):
    with pm.Model() as model:

        # Observed data (choices & rewards)
        rewards = T.as_tensor_variable(rewards)
        choices = T.as_tensor_variable(choices)
        ages = T.as_tensor_variable(ages)

        # Population-level priors (as un-informative as possible)
        priors = get_population_level_priors()
        slopes = get_slopes(fit_age_slopes)
        [eps_mu, eps_sd, beta_mu, beta_sd] = [priors[k] for k in ['eps_mu', 'eps_sd', 'beta_mu', 'beta_sd']]
        [eps_sl, beta_sl] = [slopes[k] for k in ['eps_sl', 'beta_sl']]

        if model_name == 'Bayes':
            [p_switch_mu, p_switch_sd, p_reward_mu, p_reward_sd] = [
                priors[k] for k in ['p_switch_mu', 'p_switch_sd', 'p_reward_mu', 'p_reward_sd']]
            [p_switch_sl, p_reward_sl] = [slopes[k] for k in ['p_switch_sl', 'p_reward_sl']]

        elif model_name == 'RL':
            alpha_mu, alpha_sd, calpha_sc_mu, calpha_sc_sd = [
                priors[k] for k in ['alpha_mu', 'alpha_sd', 'calpha_sc_mu', 'calpha_sc_sd']]
            alpha_sl, calpha_sl = [slopes[k] for k in ['alpha_sl', 'calpha_sl']]

        # Individual parameters (bounds avoid initial energy==-inf due to logp(RV)==-inf)
        eps = pm.Bound(pm.Normal, lower=0, upper=1)('eps', mu=eps_mu + ages * eps_sl, sd=eps_sd, shape=n_subj)
        beta = pm.Bound(pm.Normal, lower=0)('beta', mu=beta_mu + ages * beta_sl, sd=beta_sd, shape=n_subj)

        if model_name == 'Bayes':
            p_switch = pm.Bound(pm.Normal, lower=0, upper=1
                                )('p_switch', mu=p_switch_mu + ages * p_switch_sl, sd=p_switch_sd, shape=n_subj)
            p_reward = pm.Bound(pm.Normal, lower=0, upper=1
                                )('p_reward', mu=p_reward_mu + ages * p_reward_sl, sd=p_reward_sd, shape=n_subj)
            p_noisy = 1e-5 * T.ones(n_subj)

        elif model_name == 'RL':
            alpha = pm.Bound(pm.Normal, lower=0, upper=1
                             )('alpha', mu=alpha_mu + ages * alpha_sl, sd=alpha_sd, shape=n_subj)
            calpha_sc = pm.Bound(pm.Normal, lower=0, upper=1
                                 )('calpha_sc', mu=calpha_sc_mu + ages * calpha_sl, sd=alpha_sd, shape=n_subj)
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)

        # Run the model
        if model_name == 'Bayes':

            # Get likelihoods
            lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

            # Get posterior, calculate probability of subsequent trial, add eps noise
            p_right = 0.5 * T.ones(n_subj)
            p_right, _ = theano.scan(fn=post_from_lik,
                                     sequences=[lik_cor, lik_inc],
                                     outputs_info=[p_right],
                                     non_sequences=[p_switch, eps])

        elif model_name == 'RL':

            # Calculate Q-values
            Q_left, Q_right = 0.5 * T.ones(n_subj), 0.5 * T.ones(n_subj)
            [Q_left, Q_right], _ = theano.scan(fn=update_Q,
                                               sequences=[rewards, choices],
                                               outputs_info=[Q_left, Q_right],
                                               non_sequences=[alpha, calpha])

            # Translate Q-values into probabilities and add eps noise
            p_right = p_from_Q(Q_left, Q_right, beta, eps)

        # Add initial p=0.5 at the beginning of p_right
        initial_p = 0.5 * T.ones((1, n_subj))
        p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)

        # Use Bernoulli to sample responses
        model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)

        # Check model logp and RV logps (will crash if they are nan or -inf)
        if verbose or print_logps:
            print("Checking that none of the logp are -inf:")
            print("Test point: {0}".format(model.test_point))
            print("\tmodel.logp(model.test_point): {0}".format(model.logp(model.test_point)))
            for RV in model.basic_RVs:
                print("\tlogp of {0}: {1}".format(RV.name, RV.logp(model.test_point)))

        # Sample the model
        trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores)

    # Results
    model.name = model_name
    model_dict.update({model: trace})
    pm.traceplot(trace)
    model_summary = pm.summary(trace)

    # Save results
    print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model_name))

    plt.savefig(save_dir + 'plot_' + save_id + model_name + '.png')
    with open(save_dir + save_id + model_name + '.pickle', 'wb') as handle:
        pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

# Compare WAIC scores
print("Comparing models...")
pm.compareplot(pm.compare(model_dict))
plt.savefig(save_dir + 'compareplot_WAIC' + save_id + '.png')
