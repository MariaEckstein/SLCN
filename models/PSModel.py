import pickle

import theano
from theano.printing import pydotprint
import theano.tensor as T
import matplotlib.pyplot as plt
import pymc3 as pm

from shared_modeling_simulation import *
from modeling_helpers import *

# par_a_a ~ U[0,20] instead of U[0,10]?
# sigmoid: tt.nnet.sigmoid(est)/11
# Beta prior on p_switch & p_reward:
# plt.hist(pm.Beta.dist(mu=0.3, sd=0.3).random(size=int(1e4)))  => heavily skewed to the left
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.1).random(size=1e4))  => looks like normal
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.29).random(size=1e4))  => looks like uniform


# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'albenalcal_matt'
model_names = ('RL', '')

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 100
n_tune = 100
n_cores = 1
n_chains = 1

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
        # eps_a_a = pm.Uniform('eps_a_a', lower=0, upper=upper)
        # eps_a_sd = pm.Uniform('eps_a_sd', lower=0, upper=upper)
        # eps_sd_a = pm.Uniform('eps_sd_a', lower=0, upper=upper)
        # eps_sd_sd = pm.Uniform('eps_sd_sd', lower=0, upper=upper)
        # eps_a = pm.Gamma('eps_a', alpha=eps_a_a, sd=eps_a_sd, shape=n_groups)
        # eps_sd = pm.Gamma('eps_sd', alpha=eps_sd_a, sd=eps_sd_sd, shape=n_groups)
        # eps = pm.Beta('eps', alpha=eps_a[group], sd=eps_sd[group], shape=n_subj)
        eps = T.as_tensor_variable(0)
        #
        beta_mu_mu = pm.Uniform('beta_mu_mu', lower=0, upper=20, testval=5)
        beta_mu_sd = pm.Uniform('beta_mu_sd', lower=0, upper=20, testval=5)
        beta_sd_mu = pm.Uniform('beta_sd_mu', lower=0, upper=20, testval=5)
        beta_sd_sd = pm.Uniform('beta_sd_sd', lower=0, upper=20, testval=5)
        beta_mu = pm.Gamma('beta_mu', mu=beta_mu_mu, sd=beta_mu_sd, shape=n_groups, testval=5 * T.ones(n_groups))
        beta_sd = pm.Gamma('beta_sd', mu=beta_sd_mu, sd=beta_sd_sd, shape=n_groups, testval=2 * T.ones(n_groups))
        beta = pm.Gamma('beta', mu=beta_mu[group], sd=beta_sd[group], shape=n_subj, testval=5 * T.ones(n_subj))
        # beta_matt = pm.Normal('beta_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([-0.5, 0.5]))
        # beta = pm.Deterministic('beta', beta_mu[group] + beta_matt * beta_sd[group])
        # beta = T.as_tensor_variable(0)

        # beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
        # beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
        # beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])

        if model_name == 'Bayes':

            p_switch_mu_mu = pm.Uniform('p_switch_mu_mu', lower=0, upper=1, testval=0.02)
            p_switch_mu_sd = pm.Uniform('p_switch_mu_sd', lower=0, upper=1, testval=0.02)
            p_switch_sd_mu = pm.Uniform('p_switch_sd_mu', lower=0, upper=1, testval=0.02)
            p_switch_sd_sd = pm.Uniform('p_switch_sd_sd', lower=0, upper=1, testval=0.02)
            p_switch_mu = pm.Beta('p_switch_mu', mu=p_switch_mu_mu, sd=p_switch_mu_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            p_switch_sd = pm.Beta('p_switch_sd', mu=p_switch_sd_mu, sd=p_switch_sd_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            p_switch = pm.Beta('p_switch', mu=p_switch_mu[group], sd=p_switch_sd[group], shape=n_subj, testval=0.02 * T.ones(n_subj))

            p_reward_mu_mu = pm.Uniform('p_reward_mu_mu', lower=0, upper=1, testval=0.02)
            p_reward_mu_sd = pm.Uniform('p_reward_mu_sd', lower=0, upper=1, testval=0.02)
            p_reward_sd_mu = pm.Uniform('p_reward_sd_mu', lower=0, upper=1, testval=0.02)
            p_reward_sd_sd = pm.Uniform('p_reward_sd_sd', lower=0, upper=1, testval=0.02)
            p_reward_mu = pm.Beta('p_reward_mu', mu=p_reward_mu_mu, sd=p_reward_mu_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            p_reward_sd = pm.Beta('p_reward_sd', mu=p_reward_sd_mu, sd=p_reward_sd_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            p_reward = pm.Beta('p_reward', mu=p_reward_mu[group], sd=p_reward_sd[group], shape=n_subj, testval=0.02 * T.ones(n_subj))

            # p_noisy_mu_mu = pm.Uniform('p_noisy_mu_mu', lower=0, upper=1)
            # p_noisy_mu_sd = pm.Uniform('p_noisy_mu_sd', lower=0, upper=1)
            # p_noisy_sd_mu = pm.Uniform('p_noisy_sd_mu', lower=0, upper=1)
            # p_noisy_sd_sd = pm.Uniform('p_noisy_sd_sd', lower=0, upper=1)
            # p_noisy_mu = pm.Beta('p_noisy_mu', mu=p_noisy_mu_mu, sd=p_noisy_mu_sd, shape=n_groups)
            # p_noisy_sd = pm.Beta('p_noisy_sd', mu=p_noisy_sd_mu, sd=p_noisy_sd_sd, shape=n_groups)
            # p_noisy = pm.Beta('p_noisy', mu=p_noisy_mu[group], sd=p_noisy_sd[group], shape=n_subj)
            p_noisy = 1e-5 * T.ones(n_subj)

        elif model_name == 'RL':

            alpha_mu_mu = pm.Uniform('alpha_mu_mu', lower=0, upper=1, testval=0.02)
            alpha_mu_sd = pm.Uniform('alpha_mu_sd', lower=0, upper=1, testval=0.02)
            alpha_sd_mu = pm.Uniform('alpha_sd_mu', lower=0, upper=1, testval=0.02)
            alpha_sd_sd = pm.Uniform('alpha_sd_sd', lower=0, upper=1, testval=0.02)
            alpha_mu = pm.Beta('alpha_mu', mu=alpha_mu_mu, sd=alpha_mu_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            alpha_sd = pm.Beta('alpha_sd', mu=alpha_sd_mu, sd=alpha_sd_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            alpha = pm.Beta('alpha', mu=alpha_mu[group], sd=alpha_sd[group], shape=n_subj, testval=0.02 * T.ones(n_subj))
            # alpha_matt = pm.Normal('alpha_matt', mu=0, sd=1, shape=n_subj, testval=np.zeros(n_subj))
            # alpha = pm.Deterministic('alpha', alpha_mu[group] + alpha_matt * alpha_sd[group])

            # alpha_mu_diff01 = pm.Deterministic('alpha_mu_diff01', alpha_mu[0] - alpha_mu[1])
            # alpha_mu_diff02 = pm.Deterministic('alpha_mu_diff02', alpha_mu[0] - alpha_mu[2])
            # alpha_mu_diff12 = pm.Deterministic('alpha_mu_diff12', alpha_mu[1] - alpha_mu[2])

            nalpha_mu_mu = pm.Uniform('nalpha_mu_mu', lower=0, upper=1, testval=0.02)
            nalpha_mu_sd = pm.Uniform('nalpha_mu_sd', lower=0, upper=1, testval=0.02)
            nalpha_sd_mu = pm.Uniform('nalpha_sd_mu', lower=0, upper=1, testval=0.02)
            nalpha_sd_sd = pm.Uniform('nalpha_sd_sd', lower=0, upper=1, testval=0.02)
            nalpha_mu = pm.Beta('nalpha_mu', mu=nalpha_mu_mu, sd=nalpha_mu_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            nalpha_sd = pm.Beta('nalpha_sd', mu=nalpha_sd_mu, sd=nalpha_sd_sd, shape=n_groups, testval=0.02 * T.ones(n_groups))
            nalpha = pm.Beta('nalpha', mu=nalpha_mu[group], sd=nalpha_sd[group], shape=n_subj, testval=0.02 * T.ones(n_subj))
            # nalpha_matt = pm.Normal('nalpha_matt', mu=0, sd=1, shape=n_subj, testval=np.zeros(n_subj))
            # nalpha = pm.Deterministic('nalpha', nalpha_mu[group] + nalpha_matt * nalpha_sd[group])
            # nalpha = pm.Deterministic('nalpha', alpha.copy())

            # nalpha_mu_diff01 = pm.Deterministic('nalpha_mu_diff01', nalpha_mu[0] - nalpha_mu[1])
            # nalpha_mu_diff02 = pm.Deterministic('nalpha_mu_diff02', nalpha_mu[0] - nalpha_mu[2])
            # nalpha_mu_diff12 = pm.Deterministic('nalpha_mu_diff12', nalpha_mu[1] - nalpha_mu[2])

            # calpha_sc_mu_mu = pm.Uniform('calpha_sc_mu_mu', lower=0, upper=1)
            # calpha_sc_mu_sd = pm.Uniform('calpha_sc_mu_sd', lower=0, upper=1)
            # calpha_sc_sd_mu = pm.Uniform('calpha_sc_sd_mu', lower=0, upper=1)
            # calpha_sc_sd_sd = pm.Uniform('calpha_sc_sd_sd', lower=0, upper=1)
            # calpha_sc_mu = pm.Beta('calpha_sc_mu', mu=calpha_sc_mu_mu, sd=calpha_sc_mu_sd, shape=n_groups, testval=0.8 * np.ones(n_groups))
            # calpha_sc_sd = pm.Beta('calpha_sc_sd', mu=calpha_sc_sd_mu, sd=calpha_sc_sd_sd, shape=n_groups, testval=0.1 * np.ones(n_groups))
            # # calpha_sc = pm.Beta('calpha_sc', mu=calpha_sc_mu[group], sd=calpha_sc_sd[group], shape=n_subj)
            # calpha_sc_matt = pm.Normal('calpha_sc_matt', mu=0, sd=1, shape=n_subj, testval=np.zeros(n_subj))
            # calpha_sc = pm.Deterministic('calpha_sc', calpha_sc_mu[group] + calpha_sc_matt * calpha_sc_sd[group])
            calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(0))
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)

            # calpha_sc_mu_diff01 = pm.Deterministic('calpha_sc_mu_diff01', calpha_sc_mu[0] - calpha_sc_mu[1])
            # calpha_sc_mu_diff02 = pm.Deterministic('calpha_sc_mu_diff02', calpha_sc_mu[0] - calpha_sc_mu[2])
            # calpha_sc_mu_diff12 = pm.Deterministic('calpha_sc_mu_diff12', calpha_sc_mu[1] - calpha_sc_mu[2])
            #
            # cnalpha_sc_mu_mu = pm.Uniform('cnalpha_sc_mu_mu', lower=0, upper=1)
            # cnalpha_sc_mu_sd = pm.Uniform('cnalpha_sc_mu_sd', lower=0, upper=1)
            # cnalpha_sc_sd_mu = pm.Uniform('cnalpha_sc_sd_mu', lower=0, upper=1)
            # cnalpha_sc_sd_sd = pm.Uniform('cnalpha_sc_sd_sd', lower=0, upper=1)
            # cnalpha_sc_mu = pm.Beta('cnalpha_sc_mu', mu=cnalpha_sc_mu_mu, sd=cnalpha_sc_mu_sd, shape=n_groups, testval=0.8 * np.ones(n_groups))
            # cnalpha_sc_sd = pm.Beta('cnalpha_sc_sd', mu=cnalpha_sc_sd_mu, sd=cnalpha_sc_sd_sd, shape=n_groups, testval=0.1 * np.ones(n_groups))
            # cnalpha_sc_matt = pm.Normal('cnalpha_sc_matt', mu=0, sd=1, shape=n_subj, testval=np.zeros(n_subj))
            # cnalpha_sc = pm.Deterministic('cnalpha_sc', cnalpha_sc_mu[group] + cnalpha_sc_matt * cnalpha_sc_sd[group])
            cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())
            cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

            # cnalpha_sc_mu_diff01 = pm.Deterministic('cnalpha_sc_mu_diff01', cnalpha_sc_mu[0] - cnalpha_sc_mu[1])
            # cnalpha_sc_mu_diff02 = pm.Deterministic('cnalpha_sc_mu_diff02', cnalpha_sc_mu[0] - cnalpha_sc_mu[2])
            # cnalpha_sc_mu_diff12 = pm.Deterministic('cnalpha_sc_mu_diff12', cnalpha_sc_mu[1] - cnalpha_sc_mu[2])

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
