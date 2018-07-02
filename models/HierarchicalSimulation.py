import numpy as np
import pandas as pd
import scipy.stats as stats

import os

from ps_task import Task


# Function definitions
def update_Q(reward, choice, Q_old, alpha):
    return Q_old + choice * alpha * (reward - Q_old)


def p_from_Q(Q_left, Q_right, beta, epsilon):
    p_left = 1 / (1 + np.exp(-beta * (Q_left - Q_right)))  # translate Q-values into probabilities using softmax
    return epsilon * 0.5 + (1 - epsilon) * p_left  # add epsilon noise


# Switches for this script
verbose = True
n_trials = 10
n_subj = 2
save_dir = 'C:/Users/maria/MEGAsync/SLCN/PSGenRecHierarchical/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Population-level priors
alpha_mu, alpha_sd = 0.25, 0.1
beta_mu, beta_sd = 5, 3
epsilon_mu, epsilon_sd = 0.25, 0.1

alpha_lower, alpha_upper = 0, 1
beta_lower, beta_upper = 0, 50
epsilon_lower, epsilon_upper = 0, 1

# Individual parameters (drawn from truncated normal with parameters above)
alpha = stats.truncnorm(alpha_lower - alpha_mu / alpha_sd, (alpha_upper - alpha_mu) / alpha_sd,
                        loc=alpha_mu, scale=alpha_sd).rvs(n_subj)
beta = stats.truncnorm(beta_lower - beta_mu / beta_sd, (beta_upper - beta_mu) / beta_sd,
                       loc=beta_mu, scale=beta_sd).rvs(n_subj)
epsilon = stats.truncnorm(epsilon_lower - epsilon_mu / epsilon_sd, (epsilon_upper - epsilon_mu) / epsilon_sd,
                          loc=epsilon_mu, scale=epsilon_sd).rvs(n_subj)

Q_left = 0.499 * np.ones((n_subj))
Q_right = 0.499 * np.ones((n_subj))

if verbose:
    print("Alphas: {0}".format(alpha.round(2)))
    print("Betas: {0}".format(beta.round(2)))
    print("Epsilons: {0}\n".format(epsilon.round(2)))

# Set up data frames
rewards = np.zeros((n_trials, n_subj))
right_choices = np.zeros(rewards.shape)
Qs_left = np.zeros(rewards.shape)
Qs_right = np.zeros(rewards.shape)
ps_left = np.zeros(rewards.shape)

# Play task
for trial in range(n_trials):

    # Translate Q-values into action probabilities, make a choice, obtain reward, update Q-values
    p_left = p_from_Q(Q_left, Q_right, beta, epsilon)
    choice_left = np.random.binomial(n=1, p=p_left)
    choice_right = 1 - choice_left
    reward = np.array(choice_right == 1, dtype=int)  # box 1 is ALWAYS magical (for now)
    Q_left = update_Q(reward, choice_left, Q_left, alpha)
    Q_right = update_Q(reward, choice_right, Q_right, alpha)

    if verbose:
        print("\tTRIAL {0}".format(trial))
        print("Choice right:", choice_right)
        print("Reward:", reward)
        print("Q_left:", Q_left.round(3))
        print("Q_right:", Q_right.round(3))
        print("p_left:", p_left.round(2))

    # Store trial data
    ps_left[trial] = p_left
    right_choices[trial] = choice_right
    rewards[trial] = reward
    Qs_left[trial] = Q_left
    Qs_right[trial] = Q_right

# Save data
for sID in range(n_subj):

    # Create pandas DataFrame
    subj_data = pd.DataFrame()
    subj_data["selected_box"] = right_choices[:, sID]
    subj_data["reward"] = rewards[:, sID]
    subj_data["p_left"] = ps_left[:, sID]
    subj_data["Q_left"] = Qs_left[:, sID]
    subj_data["Q_right"] = Qs_right[:, sID]
    subj_data["alpha"], subj_data["beta"], subj_data["epsilon"] = alpha[sID], beta[sID], epsilon[sID]

    # Save to disc
    subj_data.to_csv(save_dir + "PS_simple_flat" + str(sID) + ".csv")
