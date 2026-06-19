import pandas as pd
import numpy as np


def get_times_non_epoched(times, epochs_times):
    # times \in (n_trials, n_time_points_per_trial, 1)
    # epochs_times \in n_trials
    # return \in n_trials * n_time_points_per_trial
    times_non_epoched = epochs_times[0] + times[0, :, 0]
    for r in range(1, len(epochs_times)):
        times_non_epoched = np.append(times_non_epoched,
                                      epochs_times[r] + times[r, :, 0])
    return times_non_epoched


def get_latents_non_epoched(latents):
    # latents \in (n_trials, n_time_points_per_trial, n_latents)
    # return \in (n_trials * n_time_points_per_trial, n_latents)
    n_trials = latents.shape[0]
    latents_non_epoched = latents[0, :, :]
    for r in range(1, n_trials):
        latents_non_epoched = np.vstack((latents_non_epoched, latents[r, :, :]))
    return latents_non_epoched


def get_equispaced_latents_non_epoched(latents):
    # latents \in (n_trials, n_time_points_per_trial, n_latents)
    # return \in (n_trials * n_time_points_per_trial, n_latents)
    n_trials = len(latents)
    latents_non_epoched = latents[0]
    for r in range(1, n_trials):
        latents_non_epoched = np.vstack((latents_non_epoched, latents[r]))
    return latents_non_epoched


def running_correlation(test_pattern, time_series):
    """
    Computes the sliding dot product between a template and a longer signal.

    This function iterates through the time_series, calculating the correlation
    between the test_pattern and a window of the signal at each step. In the
    context of neural replay, this acts as a matched filter to detect
    stereotypical latent trajectories.

    :param test_pattern: The template signal to search for (e.g., behavioral latent).
    :type test_pattern: array_like
    :param time_series: The continuous signal to search within (e.g., sleep block).
    :type time_series: array_like
    :return: The dot product scores for every valid alignment.
    :rtype: ndarray
    :raises ValueError: If test_pattern is longer than time_series.
    """
    N = len(time_series)
    L = len(test_pattern)

    if L > N:
        raise ValueError("test_pattern cannot be longer than time_series.")

    num_scores = N - L + 1
    answer = np.empty(num_scores)

    for i in range(num_scores):
        answer[i] = np.corrcoef(time_series[i:i + L], test_pattern)[0, 1]

    return answer


def running_KL(test_pattern_mean, test_pattern_var, time_series_mean,
               time_series_var):
    N = len(time_series_mean)
    L = len(test_pattern_mean)

    if L > N:
        raise ValueError("test_pattern cannot be longer than time_series.")

    num_scores = N - L + 1
    answer = np.empty(num_scores)

    for i in range(num_scores):
        answer[i] = kl_diagonal_gaussians(
						mu0=test_pattern_mean,
                        var0=test_pattern_var,
                        mu1=time_series_mean[i:i + L],
                        var1=time_series_var[i:i + L])

    return answer


def kl_diagonal_gaussians(mu0, var0, mu1, var1):
    """
    Computes the KL divergence D_KL(P || Q) between two multivariate 
    Gaussians with diagonal covariances.
    P = N(mu0, diag(var0))
    Q = N(mu1, diag(var1))
    """
    # Number of dimensions/timepoints
    K = len(mu0) 

    # 1. Ratio of variances
    term1 = var0 / var1

    # 2. Squared difference of means scaled by target variance
    term2 = (mu1 - mu0)**2 / var1

    # 3. Logarithm of the ratio of variances: ln(var1) - ln(var0)
    term3 = np.log(var1) - np.log(var0)

    # The formula: 0.5 * sum( (var0/var1) + ((mu1-mu0)^2 / var1) - 1 + ln(var1/var0) )
    # Note: sum(-1) across K dimensions is the same as subtracting K
    answer = 0.5 * np.sum(term1 + term2 - 1 + term3)

    return answer

def subset_trials_ids_data(selected_trials_ids, trials_ids, spikes_times,
                           trials_start_times, trials_end_times, epochs_times):
    indices = np.nonzero(np.in1d(trials_ids, selected_trials_ids))[0]
    spikes_times_subset = [spikes_times[i] for i in indices]
    trials_start_times_subset = trials_start_times[indices]
    trials_end_times_subset = trials_end_times[indices]
    epochs_times_subset = epochs_times[indices]

    return spikes_times_subset, trials_start_times_subset, \
            trials_end_times_subset, epochs_times_subset

def subset_clusters_data(selected_clusters, clusters,
                             spikes_times):
    indices = np.nonzero(np.in1d(clusters, selected_clusters))[0]
    n_trials = len(spikes_times)
    spikes_times_subset = [[spikes_times[r][i] for i in indices]
                           for r in range(n_trials)]

    return spikes_times_subset

def getNeuronSpikesTimesAndRegion(cluster_id, spikes_times, clusters_ids,
                                  regions):
    n_trials = len(spikes_times)
    cluster_index = clusters_ids.index(cluster_id)
    region = regions[cluster_index]
    neuron_spikes_times = [None for r in range(n_trials)]
    for r in range(n_trials):
        neuron_spikes_times[r] = spikes_times[r][cluster_index]
    return neuron_spikes_times, region

def findCorrectSequencesStartAndEndIndices(perfect_sequence, transitions_data,
                                           start_port_colname="Start_Port"):
    n_transitions = transitions_data.shape[0]
    perfect_sequence0_int = int(perfect_sequence[0])
    possible_start_indices = transitions_data.index[transitions_data[start_port_colname] == perfect_sequence0_int].tolist()

    correct_sequences_start_and_end_indices = []
    tdi = possible_start_indices[0] # tdi == transition_data_index
    del possible_start_indices[0]
    while tdi < n_transitions - 1:
        csi = 0 # csi == correct_sequence_index
        transition_sequence_start_index = tdi
        while csi < len(perfect_sequence) and tdi < n_transitions:
            # at this point correct_sequence[csi] ==
            # transitions_data.loc[tdi, start_port_colname]
            tdi += 1
            # ignore repetitions of correct sequence elements
            while tdi < n_transitions and \
                  transitions_data.loc[tdi, start_port_colname] == \
                    int(perfect_sequence[csi]):
                tdi += 1
            csi += 1
            # if all elements of correct sequence are matched
            # add a tuple to correct_sequences_start_and_end_indices
            if csi == len(perfect_sequence):
                transition_sequence_end_index = tdi - 1
                correct_sequences_start_and_end_indices.append(
                    (transition_sequence_start_index,
                     transition_sequence_end_index)
                )
            # if the next transition_data does not match the next element of
            # the sequence, start searching again from the next possible_start
            #_indices
            if tdi < n_transitions and \
               (csi == len(perfect_sequence) or \
                transitions_data.loc[tdi, start_port_colname] !=
                    int(perfect_sequence[csi])):
                while len(possible_start_indices) > 0 and \
                        possible_start_indices[0] < tdi:
                    del possible_start_indices[0]
                if len(possible_start_indices) > 0:
                    tdi = possible_start_indices[0]
                else:
                    tdi = n_transitions
                break
    return correct_sequences_start_and_end_indices


def findTrialsIDsOfPerfectSequences(perfect_sequence, transitions_data,
                                    start_port_colname="Start_Port",
                                    trial_id_colname="Trial_id"):
    n_transitions = transitions_data.shape[0]
    trials_ids = transitions_data[trial_id_colname].unique()

    perfect_trials_ids = []
    for trial_id in trials_ids:
        # Let's look to the transition_data indicices where the trials starts
        tdi = np.where(transitions_data[trial_id_colname] == trial_id)[0][0]
        psi = 0 # psi == perfect_sequence_index
        # Let's advance on the transitions and perfect_sequence while they
        # match
        while psi < len(perfect_sequence) and \
              tdi < n_transitions and \
              transitions_data.loc[tdi, trial_id_colname] == trial_id and \
              transitions_data.loc[tdi, start_port_colname] == int(perfect_sequence[psi]):
            tdi += 1
            psi += 1
        # We have a correct trial if we advnaced until the end of the sequence
        # and (we reached the end of the transitions or the current trial_id of
        # the transtions is different from that in the previous sequence
        if psi == len(perfect_sequence) and \
           (tdi == n_transitions or \
            transitions_data.loc[tdi, trial_id_colname] != trial_id):
            perfect_trials_ids.append(transitions_data.loc[tdi-1, trial_id_colname])
    return perfect_trials_ids


def buildMarkedEventsInfo(trials_timing_info, trials_indices,
                           port_numbers=np.array((2, 1, 6, 3, 7)),
                           port_colors=np.array(("orange", "red", "green",
                                                 "blue", "black")),
                           stages=np.array(["IN", "OUT"]),
                           stage_markers=["cross", "circle"]):
    n_trials = len(trials_indices)
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for trial_index, r in enumerate(trials_indices):
        trial_timing_info = \
            trials_timing_info[trials_timing_info.trial == r]
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(trial_timing_info.shape[0]):
            trial_marked_events_times.append(
                trial_timing_info.iloc[i]["timestamp"],
            )

            port_number_index = np.where(
                port_numbers == trial_timing_info.iloc[i]["port"])[0].item()

            trial_marked_events_colors.append(port_colors[port_number_index])

            stage_index = np.where(
                stages == trial_timing_info.iloc[i]["stage"])[0].item()
            trial_marked_events_markers.append(stage_markers[stage_index])

        marked_events_times[trial_index] = trial_marked_events_times
        marked_events_colors[trial_index] = trial_marked_events_colors
        marked_events_markers[trial_index] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers


def buildMarkedEventsInfoFromTrialsIDs(
        trials_ids, transitions_data,
        port_numbers=np.array([1, 2, 3, 4, 5, 6, 7]),
        port_colors=np.array(["green", "red", "cyan", "yellow", "purple", "blue", "magenta"]),
        stage_markers=["cross", "circle"],
        trial_id_colname="Trial_id",
        port_in_time_colname="P1_IN_Ephys_TS",
        port_out_time_colname="P1_OUT_Ephys_TS",
        start_port_colname="Start_Port"):
    n_trials = len(trials_ids)
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for r, trial_id in enumerate(trials_ids):
        trial_transition_data = \
            transitions_data[transitions_data[trial_id_colname]==trial_id]
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(trial_transition_data.shape[0]):
            trial_marked_events_times.append(
                trial_transition_data.iloc[i][port_in_time_colname],
            )
            trial_marked_events_times.append(
                trial_transition_data.iloc[i][port_out_time_colname],
            )

            port_number_index = np.where(
                port_numbers == trial_transition_data.iloc[i][start_port_colname])[0].item()

            trial_marked_events_colors.append(port_colors[port_number_index])
            trial_marked_events_colors.append(port_colors[port_number_index])

            trial_marked_events_markers.append(stage_markers[0])
            trial_marked_events_markers.append(stage_markers[1])

        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers


def buildMarkedEventsInfoFromTrialsTimes(
        trials_start_times, trials_end_times, epochs_times, transitions_data,
        port_numbers=np.array([1, 2, 3, 4, 5, 6, 7]),
        port_colors=np.array(["green", "red", "cyan", "yellow", "purple", "blue", "magenta"]),
        stage_linetypes=np.array(["solid", "dash"]),
        stage_markers=["cross", "circle"],
        trial_id_colname="Trial_id",
        port_in_time_colname="P1_IN_Ephys_TS",
        port_out_time_colname="P1_OUT_Ephys_TS",
        start_port_colname="Start_Port"):
    n_trials = len(trials_start_times)
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    marked_events_linetypes = [None for r in range(n_trials)]
    for r in range(n_trials):
        trial_transition_data = \
            transitions_data[np.logical_and(
                epochs_times[r]+trials_start_times[r]<=transitions_data[port_in_time_colname],
                transitions_data[port_in_time_colname]<epochs_times[r]+trials_end_times[r])]
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        trial_marked_events_linetypes = []
        for i in range(trial_transition_data.shape[0]):
            trial_marked_events_times.append(
                trial_transition_data.iloc[i][port_in_time_colname],
            )
            trial_marked_events_times.append(
                trial_transition_data.iloc[i][port_out_time_colname],
            )

            port_number_index = np.where(
                port_numbers == trial_transition_data.iloc[i][start_port_colname])[0].item()

            trial_marked_events_colors.append(port_colors[port_number_index])
            trial_marked_events_colors.append(port_colors[port_number_index])

            trial_marked_events_markers.append(stage_markers[0])
            trial_marked_events_markers.append(stage_markers[1])

            trial_marked_events_linetypes.append(stage_linetypes[0])
            trial_marked_events_linetypes.append(stage_linetypes[1])

        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
        marked_events_linetypes[r] = trial_marked_events_linetypes
    return marked_events_times, marked_events_colors, marked_events_markers, marked_events_linetypes


