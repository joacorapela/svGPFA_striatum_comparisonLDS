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

import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
                        # default=33576128)
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=17)
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        default="../../metadata/trialsIDsFrom242To341.csv")
                        # default="../../metadata/trialsIDsFrom142To241.csv")
                        # default="../../metadata/trialsIDsFrom42To141.csv")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    est_init_number = args.est_init_number
    trials_ids_filename = args.trials_ids_filename
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern
    input_model_filename = args.model_filename_pattern.format(est_res_number)
    output_model_filename_pattern = args.model_filename_pattern

    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    input_metadata_filename = estim_res_metadata_filename_pattern.format(est_res_number)
    input_metadata = configparser.ConfigParser()
    input_metadata.read(input_metadata_filename)
    epoched_spikes_times_filename = input_metadata["data_params"]["epoched_spikes_times_filename"]

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        epoch_spikes_res = pickle.load(f)
    spikes_times = epoch_spikes_res["spikes_times"]
    trials_ids = epoch_spikes_res["trials_ids"]
    trials_start_times = np.array(epoch_spikes_res["trials_start_times"])
    trials_end_times = np.array(epoch_spikes_res["trials_end_times"])
    epochs_times = np.array(epoch_spikes_res["epochs_times"])

    with open(input_model_filename, "rb") as f:
        est_results = pickle.load(f)
    kernels_types = est_results["kernels_types"]
    estimation_params = est_results["estimation_params"]
    # leg_quad_points = estimation_params["ell_calculation_params"]["leg_quad_points"]
    # leg_quad_weights = estimation_params["ell_calculation_params"]["leg_quad_weights"]
    estimated_params = est_results["estimated_params"]
    # optim_params = est_results["optim_params"]
    selected_clusters = est_results["selected_clusters"]
    clusters_ids = est_results["clusters_ids"]
    # trials_ids = est_results["trials"]
    # estimation_trials_start_times = np.array(est_results["trials_start_times"])
    # estimation_trials_end_times = np.array(est_results["trials_end_times"])
    # estimation_epochs_times = np.array(est_results["epochs_times"])

    variational_mean = estimated_params["variational_mean"]
    variational_chol_vecs = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    kernels_params = estimated_params["kernels_params"]
    # ind_points_locs = estimated_params["ind_points_locs"]

    # subset selected_clusters
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

    optim_params = dict(
        n_quad=int(est_init_config["optim_params"]["n_quad"]),
        jit=bool(est_init_config["optim_params"]["in_steps_jit"]),
        maxiter=int(est_init_config["optim_params"]["in_steps_maxiter"]),
        tol=float(est_init_config["optim_params"]["in_steps_tol"]),
        max_stepsize=float(est_init_config["optim_params"]["in_steps_max_stepsize"]),
        em_tol=float(est_init_config["optim_params"]["in_steps_em_tol"]),
        max_cont_lb_below_thr=int(est_init_config["optim_params"]["in_steps_max_cont_lb_below_thr"]),
    )

    leg_quad_points, leg_quad_weights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            n_quad=optim_params["n_quad"],
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)
    del optim_params["n_quad"]

    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    n_latents = C.shape[1]
    common_n_ind_points = variational_mean.shape[2]
    n_ind_points = [common_n_ind_points] * n_latents

    # ell_calculation_params = dict(leg_quad_points=leg_quad_points,
    #                               leg_quad_weights=leg_quad_weights,
    #                              )

    ind_points_locs = svGPFA.utils.initUtils.buildEquidistantIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        n_ind_points=n_ind_points,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

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
    estim_res_config["estimation_params"] = {"est_res_number":
                                             est_res_number}

    # build output_model_save_filename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estim_res_metadata_filename = \
            estim_res_metadata_filename_pattern.format(estResNumber)
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    output_model_filename = output_model_filename_pattern.format(estResNumber)


    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)
    print(f"Saved {estim_res_metadata_filename}")

    # initialise estimation
    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=leg_quad_points, legQuadWeights=leg_quad_weights,
            reg_param=1e-5)
            # reg_param=optim_params["prior_cov_reg_param"])

    # perform estimation
    params0 = dict(
        variational_mean=variational_mean,
        variational_chol_vecs=variational_chol_vecs,
        ind_points_locs=ind_points_locs,
    )
    additional_params = dict(
        C=C,
        d=d,
        kernels_params=kernels_params,
    )
    def inferenceOptimFunc(params, additional_params):
        value = em._eval_func(
            vMean=params["variational_mean"],
            vChol=params["variational_chol_vecs"],
            C=additional_params["C"], d=additional_params["d"],
            kernels_params=additional_params["kernels_params"],
            ind_points_locs=params["ind_points_locs"],
        )
        return value

    start_time = time.time()
    res = em.maximize_jaxopt_LBFGS_in_steps(optim_func=inferenceOptimFunc,
                                            params0=params0,
                                            additional_params=additional_params,
                                            optim_params=optim_params,
                                           )
    elapsed_time = time.time() - start_time

    resultsToSave = res.copy()
    resultsToSave["estimated_params"] = resultsToSave.pop("params")
    resultsToSave["fixed_params"] = additional_params
    resultsToSave["trials_ids"] = selected_trials_ids
    resultsToSave["selected_clusters"] = selected_clusters
    resultsToSave["clusters_ids"] = clusters_ids
    resultsToSave["kernels_types"] = kernels_types
    resultsToSave["estimation_params"] = estimation_params
    resultsToSave["optim_params"] = optim_params
    resultsToSave["trials_start_times"] = trials_start_times
    resultsToSave["trials_end_times"] = trials_end_times
    resultsToSave["epochs_times"] = epochs_times

    with open(output_model_filename, "wb") as f:
        pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(output_model_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
