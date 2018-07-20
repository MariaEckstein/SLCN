import pickle

import theano
from theano.printing import pydotprint
import matplotlib.pyplot as plt

from shared_modeling_simulation import *
from modeling_helpers import *

# z-score age!!
# Beta prior on p_switch & p_reward:
# plt.hist(pm.Beta.dist(mu=0.3, sd=0.3).random(size=10000))  => heavily skewed to the left
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.1).random(size=10000))  => looks like normal
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.29).random(size=10000))  => looks like uniform


# Switches for this script
run_on_cluster = True
verbose = False
print_logps = False
file_name_suff = 'RL_alpha_beta'
model_names = ('RL', '')

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 4000
n_tune = 100
n_chains = 2
if run_on_cluster:
    n_cores = 8
else:
    n_cores = 1

# Load to-be-fitted data
n_subj, rewards, choices, ages = load_data(run_on_cluster, fitted_data_name, kids_and_teens_only, adults_only, verbose)

# Prepare things for saving
save_dir, save_id = get_save_dir_and_save_id(run_on_cluster, file_name_suff, fitted_data_name, n_samples)

# Fit models
model_dict = {}
print("Compiling models for {0} with {1} samples and {2} tuning steps...\n".format(fitted_data_name, n_samples, n_tune))

for model_name in model_names:
    if model_name == '':
        break

    with pm.Model() as model:

        # Get population-level and individual parameters
        # eps_mu = pm.Uniform('eps_mu', lower=0, upper=1)
        # eps_sd = T.as_tensor_variable(0.01)  #pm.HalfNormal('eps_sd', sd=0.3)
        # eps_sl = T.as_tensor_variable(0)  # pm.Uniform('eps_sl', lower=-1, upper=1)
        eps = T.as_tensor_variable(0)  #pm.Beta('eps', mu=eps_mu + ages * eps_sl, sd=eps_sd, shape=n_subj)  #

        beta_mu = pm.Uniform('beta_mu', lower=0, upper=20)
        beta_sd = pm.HalfNormal('beta_sd', sd=6)  #T.as_tensor_variable(2)  #
        beta_sl = T.as_tensor_variable(0)  # pm.Uniform('beta_sl', lower=-1, upper=1)
        beta = pm.Gamma('beta', mu=beta_mu + ages * beta_sl, sd=beta_sd, shape=n_subj)  #T.as_tensor_variable(0)  #

        if model_name == 'Bayes':

            p_switch_mu = pm.Uniform('p_switch_mu', lower=0, upper=1)
            p_switch_sd = pm.HalfNormal('p_switch_sd', sd=0.3)  #T.as_tensor_variable(0.2)  #
            p_switch_sl = T.as_tensor_variable(0)  # pm.Uniform('p_switch_sl', lower=-1, upper=1)
            p_switch = pm.Beta('p_switch', mu=p_switch_mu + ages * p_switch_sl, sd=p_switch_sd, shape=n_subj)

            p_reward_mu = pm.Uniform('p_reward_mu', lower=0, upper=1)
            p_reward_sd = pm.HalfNormal('p_reward_sd', sd=0.3)  #T.as_tensor_variable(0.2)  #
            p_reward_sl = T.as_tensor_variable(0)  # pm.Uniform('p_reward_sl', lower=-1, upper=1)
            p_reward = pm.Beta('p_reward', mu=p_reward_mu + ages * p_reward_sl, sd=p_reward_sd, shape=n_subj)

            # p_noisy_mu = pm.Uniform('p_noisy_mu', lower=0, upper=1)
            # p_noisy_sd = T.as_tensor_variable(0.2)  #pm.HalfNormal('p_noisy_sd', sd=0.3)  #
            # p_noisy_sl = T.as_tensor_variable(0)  # pm.Uniform('p_noisy_sl', lower=-1, upper=1)
            # p_noisy = pm.Bound(pm.Normal, lower=0, upper=1
            #                     )('p_noisy', mu=p_noisy_mu + ages * p_noisy_sl, sd=p_noisy_sd, shape=n_subj)
            p_noisy = 1e-5 * T.ones(n_subj)

        elif model_name == 'RL':

            alpha_mu = pm.Uniform('alpha_mu', lower=0, upper=1)
            alpha_sd = pm.HalfNormal('alpha_sd', sd=0.3)  #T.as_tensor_variable(0.2)  #
            alpha_sl = T.as_tensor_variable(0)  # pm.Uniform('alpha_sl', lower=-1, upper=1)
            alpha = pm.Beta('alpha', mu=alpha_mu + ages * alpha_sl, sd=alpha_sd, shape=n_subj)

            nalpha_mu = pm.Uniform('nalpha_mu', lower=0, upper=1)
            nalpha_sd = pm.HalfNormal('nalpha_sd', sd=0.3)  #T.as_tensor_variable(0.2)  #
            nalpha_sl = T.as_tensor_variable(0)  # pm.Uniform('nalpha_sl', lower=-1, upper=1)
            # nalpha = pm.Deterministic('nalpha', alpha.copy())
            nalpha = pm.Beta('nalpha', mu=nalpha_mu + ages * nalpha_sl, sd=nalpha_sd, shape=n_subj)

            calpha_sc_mu = pm.Uniform('calpha_sc_mu', lower=0, upper=1)
            calpha_sc_sd = pm.HalfNormal('calpha_sc_sd', sd=0.3)  #T.as_tensor_variable(0.2)  #
            calpha_sc_sl = T.as_tensor_variable(0)  # pm.Uniform('calpha_sc_sl', lower=-1, upper=1)
            # calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(0))
            calpha_sc = pm.Beta('calpha_sc', mu=calpha_sc_mu + ages * calpha_sc_sl, sd=calpha_sc_sd, shape=n_subj)
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)

            cnalpha_sc_mu = pm.Uniform('cnalpha_sc_mu', lower=0, upper=1)
            cnalpha_sc_sd = pm.HalfNormal('cnalpha_sc_sd', sd=0.3)  #T.as_tensor_variable(0.2)  #
            cnalpha_sc_sl = T.as_tensor_variable(0)  # pm.Uniform('cnalpha_sc_sl', lower=-1, upper=1)
            # cnalpha_sc = pm.Deterministic('cnalpha_sc', T.as_tensor_variable(0))
            cnalpha_sc = pm.Beta('cnalpha_sc', mu=cnalpha_sc_mu + ages * cnalpha_sc_sl, sd=cnalpha_sc_sd, shape=n_subj)
            cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

        # Run the model
        if model_name == 'Bayes':

            # Get likelihoods
            lik_cor, lik_inc = get_likelihoods(rewards, choices, p_reward, p_noisy)

            # Get posterior, calculate probability of subsequent trial, add eps noise
            p_right = 0.5 * T.ones(n_subj)
            p_right, _ = theano.scan(fn=post_from_lik,
                                     sequences=[lik_cor, lik_inc],
                                     outputs_info=[p_right],
                                     non_sequences=[p_switch, eps, beta])

        elif model_name == 'RL':

            # Calculate Q-values
            Q_left, Q_right = 0.5 * T.ones(n_subj), 0.5 * T.ones(n_subj)
            [Q_left, Q_right], _ = theano.scan(fn=update_Q,
                                               sequences=[rewards, choices],
                                               outputs_info=[Q_left, Q_right],
                                               non_sequences=[alpha, nalpha, calpha, cnalpha])

            # Translate Q-values into probabilities and add eps noise
            p_right = p_from_Q(Q_left, Q_right, beta, eps)

        # Add initial p=0.5 at the beginning of p_right
        initial_p = 0.5 * T.ones((1, n_subj))
        p_right = T.concatenate([initial_p, p_right[:-1]], axis=0)

        # Use Bernoulli to sample responses
        model_choices = pm.Bernoulli('model_choices', p=p_right, observed=choices)

        # Check model logp and RV logps (will crash if they are nan or -inf)
        if verbose or print_logps:
            print_logp_info(model)

        # Sample the model
        trace = pm.sample(n_samples, tune=n_tune, chains=n_chains, cores=n_cores, nuts_kwargs=dict(target_accept=.8))

    # Get results
    model.name = model_name
    model_dict.update({model: trace})
    model_summary = pm.summary(trace)

    if not run_on_cluster:
        pm.traceplot(trace)
        plt.savefig(save_dir + 'plot_' + save_id + model_name + '.png')

    # Save results
    print('Saving trace, trace plot, model, and model summary to {0}{1}...\n'.format(save_dir, save_id + model_name))
    with open(save_dir + save_id + model_name + '.pickle', 'wb') as handle:
        pickle.dump({'trace': trace, 'model': model, 'summary': model_summary},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
