import sys
import os.path
import time
import random
import jax
import numpy as np
import pickle
import argparse
import configparser

import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        # default=66612199) # 1000 iterations
                        # default=80544908) # 100 iterations
                        # default=94346696)
                        # default=556223)
                        default=54368807)
                        # default=33576128)
    parser.add_argument("--inf_init_number", help="inference init number",
                        type=int, default=20)
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spkes filename",
                        type=str,
                        default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/47205531_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/09172167_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/69007067_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/96439322_epoched_spikes_times.pickle")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        default="../../metadata/trialsIDsFrom20200To20299.csv")
                        # default="../../metadata/trialsIDsFrom20400To20499.csv")
                        # default="../../metadata/trialsIDsFrom20300To20399.csv")
                        # default="../../metadata/trialsIDsFrom20200To20299.csv")
                        # default="../../metadata/trialsIDsFrom10200To10299.csv")
                        # default="../../metadata/trialsIDsFrom10000To10099.csv")
                        # default="../../metadata/trialsIDsFrom142To241.csv")
    parser.add_argument("--inf_init_filename_pattern",
                        help="inference initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_inference_metaData.ini")
    parser.add_argument("--inf_result_metadata_filename_pattern",
                        help="inference result metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inference_metaData.ini")
    parser.add_argument("--est_result_filename_pattern",
                        help="estimation result filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--inf_result_filename_pattern",
                        help="inference result filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inf_init_number = args.inf_init_number
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    trials_ids_filename = args.trials_ids_filename
    inf_init_filename_pattern = args.inf_init_filename_pattern
    inf_result_metadata_filename_pattern = \
        args.inf_result_metadata_filename_pattern
    input_model_filename = args.est_result_filename_pattern.format(est_res_number)
    output_model_filename_pattern = args.inf_result_filename_pattern

    inf_init_filename = inf_init_filename_pattern.format(
        inf_init_number)
    inf_init = configparser.ConfigParser()
    inf_init.read(inf_init_filename)

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        epoch_spikes_res = pickle.load(f)
    spikes_times = epoch_spikes_res["spikes_times"]
    trials_ids = epoch_spikes_res["trials_ids"]
    trials_start_times = np.array(epoch_spikes_res["trials_start_times"])
    trials_end_times = np.array(epoch_spikes_res["trials_end_times"])
    epochs_times = np.array(epoch_spikes_res["epochs_times"])
    clusters_ids = epoch_spikes_res["clusters_ids"]

    # get model parameters
    with open(input_model_filename, "rb") as f:
        est_results = pickle.load(f)
    kernels_types = est_results["kernels_types"]
    estimated_params = est_results["estimated_params"]
    selected_clusters = est_results["selected_clusters"]
    estimation_params = est_results["estimation_params"]

    variational_mean = estimated_params["variational_mean"]
    variational_chol_vecs = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    kernels_params = estimated_params["kernels_params"]

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
        jit=bool(inf_init["optim_params"]["in_steps_jit"]),
        maxiter=int(inf_init["optim_params"]["in_steps_maxiter"]),
        tol=float(inf_init["optim_params"]["in_steps_tol"]),
        max_stepsize=float(inf_init["optim_params"]["in_steps_max_stepsize"]),
        em_tol=float(inf_init["optim_params"]["in_steps_em_tol"]),
        max_cont_lb_below_thr=int(inf_init["optim_params"]["in_steps_max_cont_lb_below_thr"]),
    )
    estimation_params["optim_params"] = optim_params

    n_quad = int(inf_init["optim_params"]["n_quad"])
    leg_quad_points, leg_quad_weights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            n_quad=n_quad,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)
    estimation_params["ell_calculation_params"]["leg_quad_points"] = leg_quad_points
    estimation_params["ell_calculation_params"]["leg_quad_weights"] = leg_quad_weights

    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    n_latents = C.shape[1]
    common_n_ind_points = variational_mean.shape[2]
    n_ind_points = [common_n_ind_points] * n_latents

    ind_points_locs0 = svGPFA.utils.initUtils.buildEquidistantIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        n_ind_points=n_ind_points,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times)
    estimation_params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["inducing_points_locs0"] = ind_points_locs0

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    # save estimation initial conditions
    inf_result_config = configparser.ConfigParser()
    inf_result_config["data_params"] = {
        "trials_ids": selected_trials_ids,
        "selected_clusters": selected_clusters,
        "clusters_ids": clusters_ids,
        "nLatents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    inf_result_config["inference_params"] = {"est_res_number": est_res_number,
                                             "inf_init_number": inf_init_number,
                                         }
    # build output_model_save_filename
    infPrefixUsed = True
    while infPrefixUsed:
        infResNumber = random.randint(0, 10**8)
        inf_result_metadata_filename = \
            inf_result_metadata_filename_pattern.format(infResNumber)
        if not os.path.exists(inf_result_metadata_filename):
            infPrefixUsed = False
    output_model_filename = output_model_filename_pattern.format(infResNumber)


    with open(inf_result_metadata_filename, "w") as f:
        inf_result_config.write(f)
    print(f"Saved {inf_result_metadata_filename}")

    # initialise estimation
    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=leg_quad_points, legQuadWeights=leg_quad_weights,
            reg_param=float(inf_init["optim_params"]["in_steps_max_cont_lb_below_thr"]))

    # perform estimation
    params0 = dict(
        variational_mean=variational_mean,
        variational_chol_vecs=variational_chol_vecs,
        ind_points_locs=ind_points_locs0,
        kernels_params=kernels_params,
    )
    additional_params = dict(
        C=C,
        d=d,
    )
    def inferenceOptimFunc(params, additional_params):
        value = em._eval_func(
            vMean=params["variational_mean"],
            vChol=params["variational_chol_vecs"],
            C=additional_params["C"], d=additional_params["d"],
            kernels_params=params["kernels_params"],
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
    resultsToSave["trials_start_times"] = trials_start_times
    resultsToSave["trials_end_times"] = trials_end_times
    resultsToSave["epochs_times"] = epochs_times
    resultsToSave["n_quad"] = n_quad

    with open(output_model_filename, "wb") as f:
        pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(output_model_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
