
import sys
import os.path
import time
import random
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import configparser
import cProfile

import gcnu_common.utils.neural_data_analysis
import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=10)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=15)
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=33576128)
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    args = parser.parse_args()

    n_latents = args.n_latents
    common_n_ind_points = args.common_n_ind_points
    est_res_number = args.est_res_number
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    input_model_filename = args.model_filename_pattern.format(est_res_number)

    input_metadata_filename = estim_res_metadata_filename_pattern.format(est_res_number)
    input_metadata = configparser.ConfigParser()
    input_metadata.read(input_metadata_filename)
    epoched_spikes_times_filename = input_metadata["data_params"]["epoched_spikes_times_filename"]
    est_init_number = int(input_metadata["estimation_params"]["est_init_number"])

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    trials_ids = load_res["trials_ids"]
    spikes_times = load_res["spikes_times"]
    trials_start_times = np.array(load_res["trials_start_times"])
    trials_end_times = np.array(load_res["trials_end_times"])
    epochs_times = np.array(load_res["epochs_times"])
    clusters_ids = load_res["clusters_ids"]

    with open(input_model_filename, "rb") as f:
        est_results = pickle.load(f)

    n_trials = len(est_results["trials"])
    n_clusters = est_results["estimated_params"]["C"].shape[0]
    # n_latents = est_results["estimated_params"]["C"].shape[1]
    # common_n_ind_points = est_results["estimated_params"]["ind_points_locs"].shape[2]
    # breakpoint()

    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    #    build dynamic parameter specifications
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params_spec = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #   build config file parameters specification
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params_spec = \
        svGPFA.utils.initUtils.getParamsDictFromStringsDict(
            n_latents=n_latents, n_trials=n_trials,
            strings_dict=strings_dict, args_info=args_info)
    #    build default parameter specificiations
    default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_trials=n_trials, n_latents=n_latents,
        common_n_ind_points=common_n_ind_points)
    #    finally, get the parameters from the dynamic,
    #    configuration file and default parameter specifications
    params, kernels_types, = \
        svGPFA.utils.initUtils.getParamsAndKernelsTypes(
            n_trials=n_trials, n_clusters=n_clusters, n_latents=n_latents,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times,
            dynamic_params_spec=dynamic_params_spec,
            config_file_params_spec=config_file_params_spec)
            # config_file_params_spec=config_file_params_spec,
            # default_params_spec=default_params_spec)

    est_results["leg_quad_weights"] = params["ell_calculation_params"]["leg_quad_weights"]
    est_results["leg_quad_points"] = params["ell_calculation_params"]["leg_quad_points"]

    breakpoint()

    with open(input_model_filename, "wb") as f:
        pickle.dump(est_results, f)
    print("Saved results to {:s}".format(input_model_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
