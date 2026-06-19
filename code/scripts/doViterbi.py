
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
                        type=int,
                        # default=556223)
                        default=54368807)
    parser.add_argument("--test_est_res_number",
                        help="model estimation result number used for testing the HMM model",
                        type=int,
                        default=91676545)
                        # default=96281561)
                        # default=7996538)
                        # default=57514742)
                        # default=71005668)
                        # default=87796368)
                        # default=556223)
                        # type=int, default=20263319)
                        # type=int, default=99226606)
                        # type=int, default=42833278)
                        # type=int, default=99749566)
                        # type=int, default=88072043)
                        # type=int, default=92418550)
                        # type=int, default=54368807)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--estimated_model_filename_pattern",
                        help="estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--inferred_model_filename_pattern",
                        help="inferred model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="hmm parameters filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params.{:s}")
    parser.add_argument("--most_prob_states_seq_filename_pattern", type=str,
                        help="filtering_res filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/train{:08d}_test{:08d}_hmm_most_prob_state_seq.pickle")
    args = parser.parse_args()

    train_est_res_number = args.train_est_res_number
    test_est_res_number = args.test_est_res_number
    inferred = args.inferred
    if inferred:
        test_model_filename = args.inferred_model_filename_pattern.format(test_est_res_number)
    else:
        test_model_filename = args.estimated_model_filename_pattern.format(test_est_res_number)
    hmm_params_metadata_filename = args.hmm_params_filename_pattern.format(
        train_est_res_number, "ini")
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        train_est_res_number, "pickle")
    most_prob_states_seq_filename = args.most_prob_states_seq_filename_pattern.format(
        train_est_res_number, test_est_res_number)

    metadata_config = configparser.ConfigParser()
    metadata_config.read(hmm_params_metadata_filename)
    latents_sample_rate = int(metadata_config["estimation_params"]["latents_sample_rate"])

    with open(test_model_filename, "rb") as f:
        test_est_results = pickle.load(f)

    kernels_types = test_est_results["kernels_types"]
    # reg_param = test_est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    reg_param = 1e-5
    estimated_params = test_est_results["estimated_params"]
    if inferred:
        fixed_params = test_est_results["fixed_params"]
        # fixed_params = test_est_results["fixed_params"][0]
    trials_start_times = test_est_results["trials_start_times"]
    trials_end_times = test_est_results["trials_end_times"]

    vMean = estimated_params["variational_mean"]
    if inferred:
        kernels_params = fixed_params["kernels_params"]
        C = fixed_params["C"]
    else:
        kernels_params = estimated_params["kernels_params"]
        C = estimated_params["C"]
    ind_points_locs = estimated_params["ind_points_locs"]

    trials_times = svGPFA.utils.miscUtils.getEquispacedTrialsTimes(
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        sample_rate=latents_sample_rate)

    # extract latents means
    l_means = svGPFA.utils.statsUtils.computeLatentsMeansWithEquispacedTrialsTimes(
        vMean=vMean, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        trials_times=trials_times, reg_param=reg_param)

    l_means = [l_mean.transpose(1, 0) for l_mean in l_means]
    ol_means = svGPFA.utils.miscUtils.orthogonalizeLatentsMeans(
        latents_means=l_means, C=C)

    with open(hmm_params_filename, "rb") as f:
        hmm_params = pickle.load(f)
    state_labels = hmm_params["states_labels"]
    Pi = hmm_params["Pi"]
    A = hmm_params["A"]
    means = hmm_params["means"]
    covs = hmm_params["covs"]

    T = len(l_means)
    Pi_for_trials = np.tile(Pi[:, np.newaxis], (1, T))

    start_time = time.time()
    p = hmmUtils.getGaussianProbabilitiesEpoched(x=ol_means, means=means,
                                                 covs=covs)
    elapsed_time = time.time() - start_time
    print(f"getGaussianProbabilitiesEpoched elapsed time={elapsed_time}")

    start_time = time.time()
    most_prob_states_seq = hmm.inference.viterbiEpoched(Pi=Pi_for_trials, p=p, A=A)
    elapsed_time = time.time() - start_time
    print(f"viterbiEpoched elapsed time={elapsed_time}")

    results = dict(state_labels=state_labels, trials_times=trials_times,
                   most_prob_states_seq=most_prob_states_seq)
    with open(most_prob_states_seq_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Most probable states saved to {most_prob_states_seq_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
