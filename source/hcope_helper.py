import numpy as np
from scipy.stats import t
def pdis(Dx, pi_e, pi_b, gamma = 1.0):
    """
        run per-decision importance sampling (PDIS) estimator

        returns: PDIS(Dx, pi_e, pi_b)
    """
    ret = 0.0

    n = len(Dx)
    for i in range(n):  # Loop through all episodes in data D
        H_b = Dx[i]
        L = H_b.episodeLength
        ep_pdis = 0.0
        for t in range(L):  # Loop through all time steps of all episode
            pie_pib = 1.0
            reward_t = H_b.rewards[t]
            for j in range(t+1):  # Loop up to step t
                state_j = H_b.states[j]
                action_j = H_b.actions[j]
                pie_pib_j = (pi_e(state_j, action_j))/(pi_b(state_j, action_j))
                pie_pib *= pie_pib_j
            step_is = np.power(gamma, t) * pie_pib * reward_t
            ep_pdis += step_is
        ret += ep_pdis
    
    ret /= n
    return ret

def _pdis_history(H_b, pi_e, pi_b, gamma = 1.0):
    """
        run per-decision importance sampling (PDIS) estimator on one history

        returns: PDIS(H_b, pi_e, pi_b)
    """
    ret = 0.0

    L = H_b.episodeLength
    ep_pdis = 0.0
    for t in range(L):  # Loop through all time steps of all episode
        pie_pib = 1.0
        reward_t = H_b.rewards[t]
        for j in range(t+1):  # Loop up to step t
            state_j = H_b.states[j]
            action_j = H_b.actions[j]
            pie_pib_j = (pi_e(state_j, action_j))/(pi_b(state_j, action_j))
            pie_pib *= pie_pib_j
        step_is = np.power(gamma, t) * pie_pib * reward_t
        ep_pdis += step_is
    ret += ep_pdis

    return ret

def _variance_of_D(D, pi_e, pi_b):
    """compute the variance of a collection of histories"""
    n = len(D)
    # Compute the mean PDIS
    sum_pdis = 0.0
    for i in range(n):
        sum_pdis += _pdis_history(D[i], pi_e, pi_b)
    mean_pdis = sum_pdis / n

    sum_pdis_diff_sq = 0.0
    for i in range(n):  # Loop through all histories
        H_i = D[i]
        diff = _pdis_history(H_i, pi_e, pi_b) - mean_pdis
        sum_pdis_diff_sq += np.power(diff ,2)

    sum_pdis_diff_sq *= 1/(n-1)

    return np.sqrt(sum_pdis_diff_sq)


def safety_test(Ds, pi_e, pi_b, Ds_size, delta, c):
    """
        Perform the safety test on pi_e with variance and pdis

        return true/false for pi_e to be returned
    """
    pdis_Ds = pdis(Ds, pi_e, pi_b)

    var_Ds = _variance_of_D(Ds, pi_e, pi_b)

    ttest_ret = t.ppf(1-delta, Ds_size-1)
    ttest_term = (var_Ds / np.sqrt(Ds_size)) * ttest_ret

    safety = pdis_Ds - ttest_term

    return (safety >= c)


def est_safety_test(Dc, pi_e, pi_b, Ds_size, delta, c):
    """
        Estimate the safety test on pi_e with double variance and pdis

        return true/false for pi_e to be returned
    """
    pdis_Dc = pdis(Dc, pi_e, pi_b)

    var_Dc = _variance_of_D(Dc, pi_e, pi_b)

    ttest_ret = t.ppf(1-delta, Ds_size-1)
    ttest_term = 2 * (var_Dc / np.sqrt(Ds_size)) * ttest_ret

    safety = pdis_Dc - ttest_term

    return (safety >= c)