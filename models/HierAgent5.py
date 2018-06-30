import numpy as np
# from update_Q import update_Q
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
import theano.tensor as T

# choices = T.vector('choices')
# rewards = T.vector('rewards')
# Q_left, update = theano.scan(fn=lambda a, b: a + b,
#                         sequences=[choices, rewards])
#
# Q_left_ = theano.function(inputs=[choices, rewards], outputs=Q_left, updates=update)

choices = np.ones(5)
rewards = np.ones(5)
# print(Q_left_(choices, rewards))

basic_model = pm.Model()
with basic_model:

    beta = pm.Uniform('beta', lower=0, upper=1)
    alpha = pm.Uniform('alpha', lower=0, upper=1)

    beta_print = T.printing.Print('beta')(beta)
    alpha_print = T.printing.Print('alpha')(alpha)

    Q_old = 0.1

    def update_Q(reward, Q_old, alpha, beta):
        return Q_old + alpha * beta * (reward - Q_old)
    # Q_left, _ = theano.scan(fn=update_Q,
    #                         sequences=[choices],
    #                         outputs_info=[Q_old],
    #                         non_sequences=[alpha, beta])

    Q_left, _ = theano.scan(fn=lambda a, b: a + b,
                            sequences=[rewards, choices])

    Q_left_print = T.printing.Print('Q_left')(Q_left)

    Q_right = 1 - Q_left

    p_left = 1 / (1 + np.exp(-beta * (Q_right - Q_left)))

    model_choices = pm.Bernoulli('model_choices', p=p_left, observed=choices)

    trace = pm.sample(1000, tune=100, chains=1, cores=1)

pm.traceplot(trace)
plt.show()
