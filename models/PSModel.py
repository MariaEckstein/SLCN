import pickle

import theano
from theano.printing import pydotprint
import theano.tensor as T
import matplotlib.pyplot as plt
import pymc3 as pm

from shared_modeling_simulation import *
from modeling_helpers import *

# Beta prior on p_switch & p_reward:
# plt.hist(pm.Beta.dist(mu=0.3, sd=0.3).random(size=int(1e4)))  => heavily skewed to the left
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.1).random(size=1e4))  => looks like normal
# plt.hist(pm.Beta.dist(mu=0.5, sd=0.29).random(size=1e4))  => looks like uniform


# Switches for this script
run_on_cluster = False
verbose = False
print_logps = False
file_name_suff = 'albenal'
model_names = ('RL', '')

# Which data should be fitted?
fitted_data_name = 'humans'  # 'humans', 'simulations'
kids_and_teens_only = False
adults_only = False

# Sampling details
n_samples = 100
n_tune = 50
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

        # Priors (crashes if testvals are not specified!)
        # beta_mu_mu = pm.Uniform('beta_mu_mu', lower=0, upper=20, testval=5)
        # beta_mu_sd = pm.Uniform('beta_mu_sd', lower=0, upper=20, testval=5)
        # beta_sd_mu = pm.Uniform('beta_sd_mu', lower=0, upper=20, testval=5)
        # beta_sd_sd = pm.Uniform('beta_sd_sd', lower=0, upper=20, testval=5)
        # beta_mu_matt = pm.Normal('beta_mu_matt', mu=0, sd=1, shape=n_groups, testval=np.random.choice([-1, 0, 1], n_groups))
        # beta_sd_matt = pm.Normal('beta_sd_matt', mu=0, sd=1, shape=n_groups, testval=np.random.choice([-1, 0, 1], n_groups))
        # beta_matt = pm.Normal('beta_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([-1, 0, 1], n_subj))
        # beta_mu = pm.Deterministic('beta_mu', beta_mu_mu + beta_mu_matt * beta_mu_sd)
        # beta_sd = pm.Deterministic('beta_sd', beta_mu_sd + beta_sd_matt * beta_sd_sd)
        # beta = pm.Deterministic('beta', beta_mu[group] + beta_matt * beta_sd[group])
        beta_mu_mu = pm.Uniform('beta_mu_mu', lower=0, upper=20, testval=5)
        beta_mu_sd = pm.Uniform('beta_mu_sd', lower=0, upper=20, testval=5)
        beta_sd_mu = pm.Uniform('beta_sd_mu', lower=0, upper=20, testval=5)
        beta_sd_sd = pm.Uniform('beta_sd_sd', lower=0, upper=20, testval=5)
        beta_mu = pm.Gamma('beta_mu', mu=beta_mu_mu, sd=beta_mu_sd, shape=n_groups, testval=np.random.choice([4, 6], n_groups))
        beta_sd = pm.Gamma('beta_sd', mu=beta_sd_mu, sd=beta_sd_sd, shape=n_groups, testval=np.random.choice([1, 2], n_groups))
        beta = pm.Gamma('beta', mu=beta_mu[group], sd=beta_sd[group], shape=n_subj)
        T.printing.Print('beta_mu')(beta_mu)
        T.printing.Print('beta_sd')(beta_sd)
        T.printing.Print('beta')(beta)

        beta_mu_diff01 = pm.Deterministic('beta_mu_diff01', beta_mu[0] - beta_mu[1])
        beta_mu_diff02 = pm.Deterministic('beta_mu_diff02', beta_mu[0] - beta_mu[2])
        beta_mu_diff12 = pm.Deterministic('beta_mu_diff12', beta_mu[1] - beta_mu[2])

        eps = T.ones_like(beta)

        if model_name == 'Bayes':

            p_switch_mu = pm.Uniform('p_switch_mu', lower=0, upper=1, shape=n_groups, testval=0.1)
            p_switch_sd = pm.Uniform('p_switch_sd', lower=0, upper=1, shape=n_groups, testval=0.1)
            p_switch_matt = pm.Normal('p_switch_matt', mu=0, sd=1, shape=n_subj,
                                   testval=np.random.choice([-0.1, 0, 0.1], n_subj))
            p_switch = pm.Deterministic('p_switch', p_switch_mu[group] + p_switch_sd[group] * p_switch_matt)

            p_reward_mu = pm.Uniform('p_reward_mu', lower=0, upper=1, shape=n_groups, testval=0.1)
            p_reward_sd = pm.Uniform('p_reward_sd', lower=0, upper=1, shape=n_groups, testval=0.1)
            p_reward_matt = pm.Normal('p_reward_matt', mu=0, sd=1, shape=n_subj,
                                   testval=np.random.choice([-0.1, 0, 0.1], n_subj))
            p_reward = pm.Deterministic('p_reward', p_reward_mu[group] + p_reward_sd[group] * p_reward_matt)

            p_noisy = 1e-5 * T.ones(n_subj)

        elif model_name == 'RL':

            # alpha_mu_mu = pm.Uniform('alpha_mu_mu', lower=0, upper=1, testval=0.8)
            # alpha_mu_sd = pm.Uniform('alpha_mu_sd', lower=0, upper=1, testval=0.8)
            # alpha_sd_mu = pm.Uniform('alpha_sd_mu', lower=0, upper=1, testval=0.8)
            # alpha_sd_sd = pm.Uniform('alpha_sd_sd', lower=0, upper=1, testval=0.8)
            # alpha_mu_matt = pm.Normal('alpha_mu_matt', mu=0, sd=1, shape=n_groups, testval=np.zeros(n_groups))
            # alpha_sd_matt = pm.Normal('alpha_sd_matt', mu=0, sd=1, shape=n_groups, testval=np.zeros(n_groups))
            # alpha_matt = pm.Normal('alpha_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([-1, 1], n_subj))
            # alpha_mu = pm.Deterministic('alpha_mu', alpha_mu_mu + alpha_mu_matt * alpha_mu_sd)
            # alpha_sd = pm.Deterministic('alpha_sd', alpha_sd_mu + alpha_sd_matt * alpha_sd_sd)
            # alpha = pm.Deterministic('alpha', alpha_mu[group] + alpha_matt * alpha_sd[group])
            alpha_mu_mu = pm.Uniform('alpha_mu_mu', lower=0, upper=1, testval=0.8)
            alpha_mu_sd = pm.Uniform('alpha_mu_sd', lower=0, upper=1, testval=0.1)
            alpha_sd_mu = pm.Uniform('alpha_sd_mu', lower=0, upper=1, testval=0.1)
            alpha_sd_sd = pm.Uniform('alpha_sd_sd', lower=0, upper=1, testval=0.01)
            alpha_mu = pm.Beta('alpha_mu', mu=alpha_mu_mu, sd=alpha_mu_sd, shape=n_groups, testval=np.random.choice([0.7, 0.8], n_groups))
            alpha_sd = pm.Beta('alpha_sd', mu=alpha_sd_mu, sd=alpha_sd_sd, shape=n_groups)
            alpha = pm.Beta('alpha', mu=alpha_mu[group], sd=alpha_sd[group], shape=n_subj)

            # nalpha_mu = pm.Uniform('nalpha_mu', lower=0, upper=1, shape=n_groups, testval=0.9)
            # nalpha_sd = pm.Uniform('nalpha_sd', lower=0, upper=1, shape=n_groups, testval=0.1)
            # nalpha_matt = pm.Normal('nalpha_matt', mu=0, sd=1, shape=n_subj, testval=np.random.choice([-1, 1], n_subj))
            # nalpha = pm.Deterministic('nalpha', nalpha_mu[group] + nalpha_sd[group] * nalpha_matt)
            nalpha = pm.Deterministic('nalpha', alpha.copy())

            # calpha_sc_mu = pm.Uniform('calpha_sc_mu', lower=0, upper=1, shape=n_groups, testval=0.1)
            # calpha_sc_sd = pm.Uniform('calpha_sc_sd', lower=0, upper=1, shape=n_groups, testval=0.1)
            # calpha_sc_matt = pm.Normal('calpha_sc_matt', mu=0, sd=1, shape=n_subj,
            #                        testval=np.random.choice([-0.1, 0, 0.1], n_subj))
            # calpha_sc = pm.Deterministic('calpha_sc', calpha_sc_mu[group] + calpha_sc_sd[group] * calpha_sc_matt)
            calpha_sc = pm.Deterministic('calpha_sc', T.as_tensor_variable(0))
            calpha = pm.Deterministic('calpha', alpha * calpha_sc)

            # cnalpha_sc_mu = pm.Uniform('cnalpha_sc_mu', lower=0, upper=1, shape=n_groups, testval=0.1)
            # cnalpha_sc_sd = pm.Uniform('cnalpha_sc_sd', lower=0, upper=1, shape=n_groups, testval=0.1)
            # cnalpha_sc_matt = pm.Normal('cnalpha_sc_matt', mu=0, sd=1, shape=n_subj,
            #                        testval=np.random.choice([-0.1, 0, 0.1], n_subj))
            # cnalpha_sc = pm.Deterministic('cnalpha_sc', cnalpha_sc_mu[group] + cnalpha_sc_sd[group] * cnalpha_sc_matt)
            cnalpha_sc = pm.Deterministic('cnalpha_sc', calpha_sc.copy())
            cnalpha = pm.Deterministic('cnalpha', nalpha * cnalpha_sc)

            # Group differences?
            alpha_mu_diff01 = pm.Deterministic('alpha_mu_diff01', alpha_mu[0] - alpha_mu[1])
            alpha_mu_diff02 = pm.Deterministic('alpha_mu_diff02', alpha_mu[0] - alpha_mu[2])
            alpha_mu_diff12 = pm.Deterministic('alpha_mu_diff12', alpha_mu[1] - alpha_mu[2])

            # nalpha_mu_diff01 = pm.Deterministic('nalpha_mu_diff01', nalpha_mu[0] - nalpha_mu[1])
            # nalpha_mu_diff02 = pm.Deterministic('nalpha_mu_diff02', nalpha_mu[0] - nalpha_mu[2])
            # nalpha_mu_diff12 = pm.Deterministic('nalpha_mu_diff12', nalpha_mu[1] - nalpha_mu[2])

            # calpha_sc_mu_diff01 = pm.Deterministic('calpha_sc_mu_diff01', calpha_sc_mu[0] - calpha_sc_mu[1])
            # calpha_sc_mu_diff02 = pm.Deterministic('calpha_sc_mu_diff02', calpha_sc_mu[0] - calpha_sc_mu[2])
            # calpha_sc_mu_diff12 = pm.Deterministic('calpha_sc_mu_diff12', calpha_sc_mu[1] - calpha_sc_mu[2])
            #
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
