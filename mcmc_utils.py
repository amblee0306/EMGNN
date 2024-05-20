import random
import math
import numpy as np
from tqdm import tqdm
import copy


def proposal_function(low, high, interval):
    N = int((high-low)/interval) + 1
    discretize_lambdas = list(np.linspace(low,high,N,endpoint=True))
    new_lambda = random.choice(discretize_lambdas)
    return new_lambda


def likelihood_fn(overall_loss_val, gamma):
    return math.exp(-gamma*overall_loss_val)


def MCMC(new_lambda, prev_lmb, GAMMA, previous_loss, current_val_loss, prev_pt):
    accepted = 0
    if prev_pt is None:
        old_posterior = likelihood_fn(previous_loss, gamma=GAMMA) # * p_lambda
        new_posterior = likelihood_fn(current_val_loss, gamma=GAMMA) # * new_lambda # for the new_lambda
    else:
        old_posterior = likelihood_fn(previous_loss, gamma=GAMMA) * prev_pt[prev_lmb]
        new_posterior = likelihood_fn(current_val_loss, gamma=GAMMA) * prev_pt[new_lambda] # for the new_lambda

    if old_posterior == 0.0:
        print('previous_loss', previous_loss, 'GAMMA', GAMMA, 'current_val_loss', current_val_loss)
        eps = 1e-15 # some times have float division zero error
        old_posterior = old_posterior + eps
    A = min(1, new_posterior / old_posterior)
    u = np.random.uniform(0,1)
    if u < A:  # accept
        accepted = 1
    else:
        pass # reject lambda
    return accepted, new_lambda, current_val_loss


def form_distr_dict(p_lambda_list):
    import collections
    total_count = len(p_lambda_list)
    counter = collections.Counter(p_lambda_list)
    new_distr_dictionary = dict(counter)
    for k,v in new_distr_dictionary.items():
        new_distr_dictionary[k] = v/total_count
    return new_distr_dictionary
