
import sys
import time
import configparser
import numpy as np
import jax.numpy as jnp
import pickle
import argparse

import svGPFA.utils.statsUtils
import hmm.inference
import hmmUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_est_res_number",
                        help="model estimation result number used for learning the HMM model",
                        type=int, default=54368807)
    parser.add_argument("--test_est_res_number",
                        help="model estimation result number used for testing the HMM model",
                        type=int, default=54368807)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--model_filename_pattern",
                        help="model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="hmm parameters filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params.npz")
    parser.add_argument("--most_prob_states_seq_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_hmm_most_prob_state_seq.npz")
    args = parser.parse_args()

    train_est_res_number = args.train_est_res_number
    test_est_res_number = args.test_est_res_number
    inferred = args.inferred
    test_model_filename = args.model_filename_pattern.format(test_est_res_number)
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        train_est_res_number)
    most_prob_states_seq_filename = args.most_prob_states_seq_filename_pattern.format(
        train_est_res_number, test_est_res_number)


    with open(test_model_filename, "rb") as f:
        test_est_results = pickle.load(f)

    kernels_types = test_est_results["kernels_types"]

    leg_quad_points = test_est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    reg_param = test_est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    estimated_params = test_est_results["estimated_params"]
    if inferred:
        fixed_params = test_est_results["fixed_params"]

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
    # l_means \in n_latents \times n_quad \times n_trials
    l_means = np.transpose(jnp.asarray(l_means), (0, 2, 1))

    load_res = np.load(hmm_params_filename)
    state_labels = load_res["states_labels"]
    Pi = load_res["Pi"]
    A = load_res["A"]
    means = load_res["means"]
    covs = load_res["covs"]

    T = l_means.shape[2]
    Pi_for_trials = np.tile(Pi[:, np.newaxis], (1, T))

    start_time = time.time()
    p = hmmUtils.getGaussianProbabilitiesEpoched(x=l_means, means=means,
                                                 covs=covs)
    elapsed_time = time.time() - start_time
    print(f"getGaussianProbabilitiesEpoched elapsed time={elapsed_time}")

    start_time = time.time()
    most_prob_states_seq = hmm.inference.viterbiEpoched(Pi=Pi_for_trials, p=p, A=A)
    elapsed_time = time.time() - start_time
    print(f"viterbiEpoched elapsed time={elapsed_time}")


    np.savez(most_prob_states_seq_filename, state_labels=state_labels, trials_times=trials_times, most_prob_states_seq=most_prob_states_seq)

    print(f"Most probable states saved to {most_prob_states_seq_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
