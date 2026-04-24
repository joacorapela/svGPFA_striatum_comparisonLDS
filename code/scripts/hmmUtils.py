import copy
import numpy as np
import scipy.stats


def findAndReplaceStates(behavioral_states, find_states, replace_states):
    '''
    :param behavioral_states: behavioral states
    :type behavioral_states: numpy array
    :param find_states: states to find
    :type find_states: list like
    :param replace_states: states to find
    :type replace_states: list like
    :return: if a state in behavioral_states equals find_states[i] then its replaced by replace_states[i]
    :rtype: numpy array
    '''

    assert(len(find_states) == len(replace_states))
    replaced_states = copy.deepcopy(behavioral_states)
    for i, find_state in enumerate(find_states):
        indices = np.where(find_state == behavioral_states)[0]
        if len(indices) > 0:
            replaced_states[indices] = replace_states[i]
    return replaced_states


def markErrors(behavioral_states,
                correct_states=[2, 1, 6, 3, 7,
                                22, 11, 66, 33, 77,
                                21, 16, 63, 37, 72],
                error_state=0):
    '''
    :param behavioral_states: behavioral states
    :type behavioral_states: numpy array
    :param correct_states: correct behavioral states
    :type correct_states: list like
    :return: behavior_states where states not belonging to correct states are replaced by error_state
    :rtype: numpy array
    '''

    corrected_states = copy.deepcopy(behavioral_states)
    for i, behavioral_state in enumerate(behavioral_states):
        if not behavioral_state in correct_states:
            corrected_states[i] = error_state
    return corrected_states


def getBehavioralStates(bins_centers, port_labels, enter_times, exit_times):
    '''
    :param bins_centers: bins centers
    :type bins_centers: list like
    :param port_labels: port labels
    :type port_labels: list like
    :param enter_times: port labels
    :type enter_times: list like
    :return: behavioral_states
    :rtype: numpy array
    '''

    behavioral_states = [None for i in range(len(bins_centers))]

    for i, t in enumerate(bins_centers):
        aux = np.where(t < enter_times)[0]
        if len(aux) > 0:
            index = aux[0]
            # check if in previous port
            if index > 0 and t < exit_times[index-1]:
                behavioral_states[i] = port_labels[index-1].item()
            # check if in transition between the previous and current port
            if index > 0 and exit_times[index-1] <= t and t < enter_times[index]:
                behavioral_states[i] = (port_labels[index-1].item() * 10 +
                                        port_labels[index].item())
    return np.array(behavioral_states)


def getFilteredBehavioralStates(bins_centers, port_labels, enter_times, exit_times):
    '''
    :param bins_centers: bins centers
    :type bins_centers: list like
    :param find_states: states to find
    :type find_states: list like
    :param replace_states: states to find
    :type replace_states: list like
    :return: a states labels integer array and an array with elements in [0, K-1]
    :rtype: tuple of numpy arrays
    '''
    behavioral_states = getBehavioralStates(bins_centers=bins_centers, port_labels=port_labels,
                                            enter_times=enter_times, exit_times=exit_times)
    corrected_behavioral_states = markErrors(behavioral_states=behavioral_states)
    replaced_corrected_behavioral_states = findAndReplaceStates(
        behavioral_states=corrected_behavioral_states,
        find_states=np.array([22, 11, 66, 33, 77]),
        replace_states=np.array([2, 1, 6, 3, 7]))
    states_labels = np.array([0, 2, 21, 1, 16, 6, 63, 3, 37, 7, 72],
                             dtype=np.int16)
    relabeled_replaced_corrected_behavioral_states = findAndReplaceStates(
        behavioral_states=replaced_corrected_behavioral_states,
        find_states=states_labels,
        replace_states=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    return states_labels, relabeled_replaced_corrected_behavioral_states


def getFilteredBehavioralStatesEpoched(trials_times, epochs_times, port_labels,
                                       enter_times, exit_times):
    n_trials = len(trials_times)
    trials_behavioral_states = [None] * n_trials
    for r in range(n_trials):
        states_labels, trials_behavioral_states[r] = getFilteredBehavioralStates(
            bins_centers=trials_times[r][:, 0] + epochs_times[r],
            port_labels=port_labels,
            enter_times=enter_times,
            exit_times=exit_times)
    return states_labels, trials_behavioral_states


def getGaussianProbabilities(x, means, covs):
    """
    Calculates the multivariate Gaussian probability density for each observation
    across all states.

    :param x: Observation vectors where D is the feature dimension and N is the
        number of samples.
    :type x: numpy.ndarray of shape (D, N)
    :param means: Mean vectors for each of the K states.
    :type means: numpy.ndarray of shape (D, 1, K)
    :param covs: Covariance matrices for each of the K states.
    :type covs: numpy.ndarray of shape (D, D, K)
    :return: Probability densities (likelihoods) for each sample and state.
    :rtype: numpy.ndarray of shape (N, K)
    """

    N = x.shape[1]
    K = means.shape[2]
    p = np.zeros((N, K))

    for k in range(K):
        # Extract the mean vector and covariance matrix for state k
        mean_k = means[:, 0, k]
        sigma_k = covs[:, :, k]

        # multivariate_normal.pdf takes the observation vector, mean, and cov
        p[:, k] = scipy.stats.multivariate_normal.pdf(x.T, mean=mean_k,
                                                      cov=sigma_k)

    return p


def getGaussianProbabilitiesEpoched(x, means, covs):
    """
    Calculates the multivariate Gaussian probability density for each epoched
    observation across all states.

    :param x: Epoched observation vectors where D is the feature dimension,
        N_t is the number of samples in trial T and T is the number of trials.
    :type x: list of length T, where x[t] is a numpy.ndarray of shape (N_t, D)
    :param means: Mean vectors for each of the K states.
    :type means: numpy.ndarray of shape (D, 1, K)
    :param covs: Covariance matrices for each of the K states.
    :type covs: numpy.ndarray of shape (D, D, K)
    :return: Probability densities (likelihoods) for each sample, state and trial.
    :rtype: list of length T, where answer[t] is numpy.ndarray of shape (N_t, K)
    """
    T = len(x)
    p = [None] * T

    for t in range(T):
        p[t] = getGaussianProbabilities(x=x[t].T, means=means, covs=covs)
    return p
