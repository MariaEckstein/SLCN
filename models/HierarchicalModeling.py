import pickle

import theano
from theano.printing import pydotprint
import matplotlib.pyplot as plt

from shared_modeling_simulation import *
from modeling_helpers import *

# z-score age!!
# Beta prior on p_switch & p_reward:
# plt.hist(pm.Beta.dist(mu=0.3, sd=0.3).random(size=int(1e4)))  => heavily skewed to the left
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.1).random(size=1e4))  => looks like normal
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.29).random(size=1e4))  => looks like uniform


# Switches for this script
run_on_cluster = True
verbose = False
print_logps = False
file_name_suff = 'RL_alpha_beta_nalpha_gamma_hyperpriors'
model_names = ('RL', '')

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 5000
n_tune = 1000
if run_on_cluster:
    n_cores = 3
else:
    n_cores = 1
n_chains = n_cores

# Load to-be-fitted data
n_subj, rewards, choices, age, group, n_groups = load_data(run_on_cluster, fitted_data_name, kids_and_teens_only, adults_only, verbose)

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
        # eps_a = pm.Uniform('eps_a', lower=0, upper=1e4)
        # eps_b = pm.Uniform('eps_b', lower=0, upper=1e4)
        # eps = pm.Beta('eps', alpha=eps_a, beta=eps_b, shape=n_subj)
        eps = T.as_tensor_variable(0)

        beta_a = pm.Gamma('beta_a', alpha=0.25, beta=0.05)  # similar to pm.Uniform('beta_a', lower=0, upper=20)
        beta_b = pm.Gamma('beta_b', alpha=0.25, beta=0.05)  # similar to pm.Uniform('beta_b', lower=0, upper=20)
        beta = pm.Gamma('beta', alpha=beta_a, beta=beta_b, shape=n_subj)
        # beta = T.as_tensor_variable(0)

        if model_name == 'Bayes':

            p_switch_a_a = pm.Uniform('p_switch_a_a', lower=0, upper=10)
            p_switch_a_b = pm.Uniform('p_switch_a_b', lower=0, upper=10)
            p_switch_b_a = pm.Uniform('p_switch_b_a', lower=0, upper=10)
            p_switch_b_b = pm.Uniform('p_switch_b_b', lower=0, upper=10)
            p_switch_a = pm.Gamma('p_switch_a', alpha=p_switch_a_a, beta=p_switch_a_b, shape=n_groups)
            p_switch_b = pm.Gamma('p_switch_b', alpha=p_switch_b_a, beta=p_switch_b_b, shape=n_groups)
            p_switch = pm.Beta('p_switch', alpha=p_switch_a[group], beta=p_switch_b[group])

            p_reward_a_a = pm.Uniform('p_reward_a_a', lower=0, upper=10)
            p_reward_a_b = pm.Uniform('p_reward_a_b', lower=0, upper=10)
            p_reward_b_a = pm.Uniform('p_reward_b_a', lower=0, upper=10)
            p_reward_b_b = pm.Uniform('p_reward_b_b', lower=0, upper=10)
            p_reward_a = pm.Gamma('p_reward_a', alpha=p_reward_a_a, beta=p_reward_a_b, shape=n_groups)
            p_reward_b = pm.Gamma('p_reward_b', alpha=p_reward_b_a, beta=p_reward_b_b, shape=n_groups)
            p_reward = pm.Beta('p_reward', alpha=p_reward_a[group], beta=p_reward_b[group])

            # p_noisy_a_a = pm.Uniform('p_noisy_a_a', lower=0, upper=10)
            # p_noisy_a_b = pm.Uniform('p_noisy_a_b', lower=0, upper=10)
            # p_noisy_b_a = pm.Uniform('p_noisy_b_a', lower=0, upper=10)
            # p_noisy_b_b = pm.Uniform('p_noisy_b_b', lower=0, upper=10)
            # p_noisy_a = pm.Gamma('p_noisy_a', alpha=p_noisy_a_a, beta=p_noisy_a_b, shape=n_groups)
            # p_noisy_b = pm.Gamma('p_noisy_b', alpha=p_noisy_b_a, beta=p_noisy_b_b, shape=n_groups)
            # p_noisy = pm.Beta('p_noisy', alpha=p_noisy_a[group], beta=p_noisy_b[group], shape=n_subj)
            p_noisy = 1e-5 * T.ones(n_subj)

        elif model_name == 'RL':

            
            alpha_a = pm.Gamma('alpha_a', alpha=0.25, beta=0.05)  # pm.Uniform('alpha_a', lower=0, upper=20)
            alpha_b = pm.Gamma('alpha_b', alpha=0.25, beta=0.05)  # pm.Uniform('alpha_b', lower=0, upper=20)
            alpha = pm.Beta('alpha', alpha=alpha_a, beta=alpha_b, shape=n_subj)

            nalpha_a = pm.Gamma('nalpha_a', alpha=0.25, beta=0.05)  # pm.Uniform('nalpha_a', lower=0, upper=20)
            nalpha_b = pm.Gamma('nalpha_b', alpha=0.25, beta=0.05)  # pm.Uniform('nalpha_b', lower=0, upper=20)
            # nalpha = pm.Deterministic('nalpha', alpha.copy())
            nalpha = pm.Beta('nalpha', alpha=nalpha_a, beta=nalpha_b, shape=n_subj)

            # calpha_sc_a = pm.Gamma('calpha_sc_a', alpha=0.25, beta=0.05)  # pm.Uniform('calpha_sc_a', lower=0, upper=20)
            # calpha_sc_b = pm.Gamma('calpha_sc_b', alpha=0.25, beta=0.05)  # pm.Uniform('calpha_sc_b', lower=0, upper=20)
            # calpha_sc = pm.Beta('calpha_sc', alpha=calpha_sc_a, beta=calpha_sc_b, shape=n_subj)
            calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(0))
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)

            # cnalpha_sc_a = pm.Uniform('cnalpha_sc_a', lower=0, upper=20)
            # cnalpha_sc_b = pm.Uniform('cnalpha_sc_b', lower=0, upper=20)
            # cnalpha_sc = pm.Beta('cnalpha_sc', alpha=cnalpha_sc_a, beta=cnalpha_sc_b, shape=n_subj)
            cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())
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
