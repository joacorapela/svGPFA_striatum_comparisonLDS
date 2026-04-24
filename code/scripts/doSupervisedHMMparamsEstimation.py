
import sys
import argparse
import configparser
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle

import svGPFA.utils.statsUtils
import hmm.learning
import hmmUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=556223)
                        # default=54368807)
                        # default=42833278)
                        # default=92418550)
                        # default=42976344)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--latents_sample_rate", help="plot sample rate", type=int,
                        default=50)
    parser.add_argument("--port_label_col_name",
                        help="column name for port label",
                        type=str, default="Start_Port")
    parser.add_argument("--port_enter_times_col_name",
                        help="column name for port enter (ephys) time",
                        type=str, default="P1_IN_Ephys_TS")
    parser.add_argument("--port_exit_times_col_name",
                        help="column name for port exit (ephys) time",
                        type=str, default="P1_OUT_Ephys_TS")
    parser.add_argument("--model_filename_pattern",
                        help="model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="hmm parameters filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params.{:s}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inferred = args.inferred
    latents_sample_rate = args.latents_sample_rate
    port_label_col_name = args.port_label_col_name
    port_enter_times_col_name = args.port_enter_times_col_name
    port_exit_times_col_name = args.port_exit_times_col_name
    model_filename = args.model_filename_pattern.format(est_res_number)
    transitions_data_filename = args.transitions_data_filename
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        est_res_number, "pickle")
    hmm_params_metadata_filename = args.hmm_params_filename_pattern.format(
        est_res_number, "ini")

    transitions_data = pd.read_csv(transitions_data_filename)

    with open(model_filename, "rb") as f:
        est_results = pickle.load(f)
    kernels_types = est_results["kernels_types"]
    epochs_times = est_results["epochs_times"]
    trials_start_times = est_results["trials_start_times"]
    trials_end_times = est_results["trials_end_times"]

    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    estimated_params = est_results["estimated_params"]
    if inferred:
        fixed_params = est_results["fixed_params"][0]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    ind_points_locs = estimated_params["ind_points_locs"]
    if inferred:
        C = fixed_params["C"]
        kernels_params = fixed_params["kernels_params"]
    else:
        C = estimated_params["C"]
        kernels_params = estimated_params["kernels_params"]

    trials_times = svGPFA.utils.miscUtils.getEquispacedTrialsTimes(
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        sample_rate=latents_sample_rate)

    # extract latents means and variances and estimated C
    l_means = svGPFA.utils.statsUtils.computeLatentsMeansWithEquispacedTrialsTimes(
        vMean=vMean, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        trials_times=trials_times, reg_param=reg_param)

    l_means = [l_mean.transpose(1, 0) for l_mean in l_means]
    ol_means = svGPFA.utils.miscUtils.orthogonalizeLatentsMeans(
        latents_means=l_means, C=C)

    port_labels = transitions_data[port_label_col_name].to_numpy()
    enter_times = transitions_data[port_enter_times_col_name].to_numpy()
    exit_times = transitions_data[port_exit_times_col_name].to_numpy()
    states_labels, states_seq = hmmUtils.getFilteredBehavioralStatesEpoched(
        trials_times=trials_times,
        epochs_times=epochs_times,
        port_labels=port_labels,
        enter_times=enter_times,
        exit_times=exit_times)
    abs_trials_times = [trials_times[r] + epochs_times[r] for r in range(len(trials_times))]
    Pi, A, means, covs = hmm.learning.supervisedLearningGaussianObservationsEpoched(
        states_seq=states_seq, observations_seq=ol_means,
        trials_times=abs_trials_times)

    metadata_config = configparser.ConfigParser()
    metadata_config["estimation_params"] = {
        "latents_sample_rate": latents_sample_rate,
    }
    with open(hmm_params_metadata_filename, "w") as f:
        metadata_config.write(f)
    print(f"Saved {hmm_params_metadata_filename}")

    results = dict(states_labels=states_labels, Pi=Pi, A=A,
                   trials_times=trials_times, means=means, covs=covs)

    with open(hmm_params_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"HMM params saved to {hmm_params_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
