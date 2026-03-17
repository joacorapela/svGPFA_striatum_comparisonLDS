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
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=17)
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=10)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=15)
    parser.add_argument("--profile",
                        help="use this option if you want to profile svGPFA.maximize()",
                        action="store_true")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filenamepattern",
                        type=str,
                        default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/96439322_epoched_spikes_times.pickle")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        default="../../metadata/trialsIDsFrom42To141.csv")
                        # default="../../metadata/trialsIDsFrom142To241.csv")
                        # default="../../metadata/trialsIDs_372_414.csv")
    parser.add_argument("--clusters_ids_filename", help="clusters ids filename",
                        type=str,
                        default="../../metadata/clustersIDs_striatum.csv")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--profiling_info_filename_pattern",
                        help="profiling information filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_profiling_info.txt")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_init_number = args.est_init_number
    n_latents = args.n_latents
    common_n_ind_points = args.common_n_ind_points
    profile = args.profile
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    trials_ids_filename = args.trials_ids_filename
    clusters_ids_filename = args.clusters_ids_filename
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern
    profiling_info_filename_pattern = args.profiling_info_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern

    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    trials_ids = load_res["trials_ids"]
    spikes_times = load_res["spikes_times"]
    trials_start_times = np.array(load_res["trials_start_times"])
    trials_end_times = np.array(load_res["trials_end_times"])
    epochs_times = np.array(load_res["epochs_times"])
    clusters_ids = load_res["clusters_ids"]

    # subset selected_clusters
    selected_clusters = np.genfromtxt(clusters_ids_filename,
                                      delimiter=",", dtype=np.uint64)
    spikes_times = striatumUtils.subset_clusters_data(
        selected_clusters=selected_clusters,
        clusters=clusters_ids,
        spikes_times=spikes_times,
    )

    # get selected_trials_ids
    selected_trials_ids = np.genfromtxt(trials_ids_filename, delimiter=',', dtype=np.uint64)
    spikes_times, trials_start_times, trials_end_times, epochs_times = \
            striatumUtils.subset_trials_ids_data(
                selected_trials_ids=selected_trials_ids,
                trials_ids=trials_ids,
                spikes_times=spikes_times,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times,
                epochs_times=epochs_times,
            )

    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])

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

    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estim_res_metadata_filename = \
            estim_res_metadata_filename_pattern.format(estResNumber)
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    modelSaveFilename = model_save_filename_pattern.format(estResNumber)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    # get estimation params
    leg_quad_weights = params["ell_calculation_params"]["leg_quad_weights"]
    leg_quad_points = params["ell_calculation_params"]["leg_quad_points"]
    qMu0 = params["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["mean"].squeeze()
    variational_chol_vecs = params["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["cholVecs"]
    C = params["initial_params"]["embedding"]["C0"]
    d = params["initial_params"]["embedding"]["d0"]
    Z0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["inducing_points_locs0"]

    # save estimation initial conditions
    estim_res_config = configparser.ConfigParser()
    estim_res_config["data_params"] = {
        "trials_ids": selected_trials_ids,
        "selected_clusters": selected_clusters,
        "clusters_ids": clusters_ids,
        "nLatents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    estim_res_config["estimation_params"] = {"est_init_number":
                                             est_init_number}
    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)
    print(f"Saved {estim_res_metadata_filename}")

    # initialise estimation
    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=leg_quad_points, legQuadWeights=leg_quad_weights,
            reg_param=params["optim_params"]["prior_cov_reg_param"],
           )

    # perform estimation
    params0 = dict(
        variational_mean=qMu0,
        variational_chol_vecs=variational_chol_vecs,
        C=C,
        d=d,
        kernels_params=kernels_params0,
        ind_points_locs=Z0,
    )

    optim_params = dict(
        jit=bool(est_init_config["optim_params"]["in_steps_jit"]),
        maxiter=int(est_init_config["optim_params"]["in_steps_maxiter"]),
        tol=float(est_init_config["optim_params"]["in_steps_tol"]),
        max_stepsize=float(est_init_config["optim_params"]["in_steps_max_stepsize"]),
        em_tol=float(est_init_config["optim_params"]["in_steps_em_tol"]),
        max_cont_lb_below_thr=int(est_init_config["optim_params"]["in_steps_max_cont_lb_below_thr"]),
    )

    if profile:
        profiling_info_filename_pattern = \
            profiling_info_filename_pattern.format(estResNumber)

    start_time = time.time()
    res = em.maximize_jaxopt_LBFGS_in_steps(
        optim_func=em._eval_func_params_as_dict, params0=params0,
        optim_params=optim_params,
    )
    elapsed_time = time.time() - start_time

    print(f"elapsed time={elapsed_time}")

    if profile:
        pr.disable()
        pr.dump_stats(filename=profiling_info_filename)

    resultsToSave = res.copy()
    resultsToSave["estimated_params"] = resultsToSave.pop("params")
    resultsToSave["trials_ids"] = selected_trials_ids
    resultsToSave["selected_clusters"] = selected_clusters
    resultsToSave["clusters_ids"] = clusters_ids
    resultsToSave["kernels_types"] = kernels_types
    resultsToSave["estimation_params"] = params
    # resultsToSave["optim_params"] = optim_params
    resultsToSave["trials_start_times"] = trials_start_times
    resultsToSave["trials_end_times"] = trials_end_times
    resultsToSave["epochs_times"] = epochs_times

    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
        print("Saved results to {:s}".format(modelSaveFilename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
