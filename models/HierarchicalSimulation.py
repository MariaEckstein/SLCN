import numpy as np
import scipy.stats as stats
# from HierarchicalModeling import update_Q, p_from_Q  # Why does it not work??


# Function definitions
def update_Q(reward, choice, Q_old, alpha):
    return Q_old + choice * alpha * (reward - Q_old)


def p_from_Q(Q_left, Q_right, beta, epsilon):
    p_left = 1 / (1 + np.exp(-beta * (Q_left - Q_right)))  # translate Q-values into probabilities using softmax
    return epsilon * 0.5 + (1 - epsilon) * p_left  # add epsilon noise


# Switches for this script
verbose = True
n_trials = 3
data_dir = 'C:/Users/maria/MEGAsync/SLCN/PSGenRecHierarchical/'
n_subj = 2

# Population-level priors (TD: get from actual fitting results)
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

if verbose:
    print("Alphas: {0}".format(alpha.round(2)))
    print("Betas: {0}".format(beta.round(2)))
    print("Epsilons: {0}\n".format(epsilon.round(2)))

# Set up data frames
rewards = np.zeros((n_trials, n_subj))
left_choices = np.zeros(rewards.shape)
Qs_left = np.zeros(rewards.shape)
ps_left = np.zeros(rewards.shape)

# Play task
Q_left = 0.499 * np.ones((n_subj))
Q_right = 0.499 * np.ones((n_subj))

for trial in range(n_trials):

    # Calculate action probabilities from Q-values, make a choice, obtain reward, update Q-values
    p_left = p_from_Q(Q_left, Q_right, beta, epsilon)
    choice = np.random.binomial(n=1, p=p_left)
    reward = np.array(choice == 1, dtype=int)  # box 1 is magical for now
    Q_left = update_Q(reward, choice, Q_left, alpha)

    if verbose:
        print("\tTRIAL {0}".format(trial))
        print("Choice:", choice)
        print("Reward:", reward)
        print("Q:", Q_left.round(3))

    # Store trial data
    ps_left[trial] = p_left
    left_choices[trial] = choice
    rewards[trial] = reward
    Qs_left[trial] = Q_left

# Save dataframes (TD)

