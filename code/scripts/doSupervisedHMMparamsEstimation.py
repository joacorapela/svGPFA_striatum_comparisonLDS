
import sys
import warnings
import configparser
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle
import argparse
import plotly.colors

import svGPFA.utils.statsUtils
import hmm.learning
import hmmUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
                        # default=92418550)
                        # default=42976344)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
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
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params.npz")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inferred = args.inferred
    port_label_col_name = args.port_label_col_name
    port_enter_times_col_name = args.port_enter_times_col_name
    port_exit_times_col_name = args.port_exit_times_col_name
    model_filename = args.model_filename_pattern.format(est_res_number)
    transitions_data_filename = args.transitions_data_filename
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        est_res_number)

    transitions_data = pd.read_csv(transitions_data_filename)

    with open(model_filename, "rb") as f:
        est_results = pickle.load(f)
    kernels_types = est_results["kernels_types"]
    epochs_times = est_results["epochs_times"]

    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    estimated_params = est_results["estimated_params"]
    if inferred:
        fixed_params = est_results["fixed_params"][0]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    if inferred:
        kernels_params = fixed_params["kernels_params"]
    else:
        kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    # extract latents means and varances and estimated C
    l_means, _ = svGPFA.utils.statsUtils.computeLatents(
        vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    # trials_times \in n_trials \times n_quad \times 1
    trials_times = jnp.asarray(leg_quad_points)
    # l_means \in n_trials \times n_latents \times n_quad
    l_means = np.transpose(jnp.asarray(l_means), (1, 0, 2))

    port_labels = transitions_data[port_label_col_name].to_numpy()
    enter_times = transitions_data[port_enter_times_col_name].to_numpy()
    exit_times = transitions_data[port_exit_times_col_name].to_numpy()
    states_labels, states_seq = hmmUtils.getFilteredBehavioralStatesEpoched(
        trials_times=trials_times,
        epochs_times=epochs_times,
        port_labels=port_labels,
        enter_times=enter_times,
        exit_times=exit_times)
    Pi, A, means, covs = hmm.learning.supervisedLearningEpochedGaussianObservations(
        states_seq=states_seq, observations_seq=l_means)

    np.savez(hmm_params_filename, states_labels=states_labels, Pi=Pi, A=A,
             bins_centers=trials_times, means=means, covs=covs)
    print(f"HMM params saved to {hmm_params_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
