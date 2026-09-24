
import sys
import time
import random
import os.path
import argparse
import configparser
import pickle
import numpy as np

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
    parser.add_argument("--speed_up_factor",
                        help=("before running the Viterbi algorithm " +
                              "resample the latents with the original " +
                              "sampling rate multiplied by this factor, " +
                              "to test if replay happens at a faster speed"),
                        type=float,
                        default=1.0)
                        # default=50.0)
    parser.add_argument("--estimated_model_filename_pattern",
                        help="estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="hmm parameters filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params.{:s}")
    parser.add_argument("--res_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_most_prob_state_seq.{:s}")
    args = parser.parse_args()

    train_est_res_number = args.train_est_res_number
    test_est_res_number = args.test_est_res_number
    speed_up_factor = args.speed_up_factor
    test_model_filename = args.estimated_model_filename_pattern.format(test_est_res_number)
    hmm_params_metadata_filename = args.hmm_params_filename_pattern.format(
        train_est_res_number, "ini")
    hmm_params_filename = args.hmm_params_filename_pattern.format(
        train_est_res_number, "pickle")
    res_filename_pattern = args.res_filename_pattern

    metadata_config = configparser.ConfigParser()
    metadata_config.read(hmm_params_metadata_filename)
    latents_sample_rate = int(metadata_config["estimation_params"]["latents_sample_rate"])

    # build res_filename
    prefixUsed = True
    while prefixUsed:
        res_number = random.randint(0, 10**8)
        metadata_filename = res_filename_pattern.format(res_number,
                                                        "metadata")
        if not os.path.exists(metadata_filename):
            prefixUsed = False
    res_filename = res_filename_pattern.format(res_number, "pickle")

    with open(test_model_filename, "rb") as f:
        test_est_results = pickle.load(f)

    kernels_types = test_est_results["kernels_types"]
    # reg_param = test_est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    reg_param = 1e-5
    estimated_params = test_est_results["estimated_params"]
    if "fixed_params" in test_est_results:
        fixed_params = test_est_results["fixed_params"]
        # fixed_params = test_est_results["fixed_params"][0]
    trials_start_times = test_est_results["trials_start_times"]
    trials_end_times = test_est_results["trials_end_times"]

    vMean = estimated_params["variational_mean"]
    if "C" in estimated_params:
        C = estimated_params["C"]
    elif "C" in fixed_params:
        C = fixed_params["C"]
    else:
        raise ValueError("kernels_params and C not found in either estimated_ or fixed_params")
    kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    trials_times = svGPFA.utils.miscUtils.getEquispacedTrialsTimes(
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        sample_rate=latents_sample_rate*speed_up_factor)

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

    metadata = configparser.ConfigParser()
    metadata["params"] = {
        "train_est_res_number": train_est_res_number,
        "test_est_res_number": test_est_res_number,
        "speed_up_factor": speed_up_factor,
        "hmm_params_filename": hmm_params_filename,
    }
    with open(metadata_filename, "w") as f:
        metadata.write(f)
    print(f"Saved {metadata_filename}")

    results = dict(state_labels=state_labels, trials_times=trials_times,
                   most_prob_states_seq=most_prob_states_seq)
    with open(res_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {res_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
