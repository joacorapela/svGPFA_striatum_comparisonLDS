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

import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
    parser.add_argument("--maxiter", help="maximum number of iterations",
                        type=int,
                        default=100000)
    parser.add_argument("--est_init_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--metadata_filename_pattern",
                        help="estimation metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--est_res_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    args = parser.parse_args()

    in_est_res_number = args.in_est_res_number
    maxiter = args.maxiter
    # est_init_number = args.est_init_number
    est_init_filename_pattern = args.est_init_filename_pattern
    # epoched_spikes_times_filename = args.epoched_spikes_times_filename
    # trials_ids_filename = args.trials_ids_filename
    metadata_filename_pattern = args.metadata_filename_pattern
    est_res_filename_pattern = args.est_res_filename_pattern

    metadata_filename = args.metadata_filename_pattern.format(in_est_res_number)
    metaData = configparser.ConfigParser()
    metaData.read(metadata_filename)
    epoched_spikes_times_filename = metaData["data_params"]["epoched_spikes_times_filename"]
    est_init_number = int(metaData["estimation_params"]["est_init_number"])

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        epoched_spikes_res = pickle.load(f)
    spikes_times = epoched_spikes_res["spikes_times"]
    trials_ids = epoched_spikes_res["trials_ids"]
    trials_start_times = np.array(epoched_spikes_res["trials_start_times"])
    trials_end_times = np.array(epoched_spikes_res["trials_end_times"])
    epochs_times = np.array(epoched_spikes_res["epochs_times"])
    clusters_ids = epoched_spikes_res["clusters_ids"]

    # get estimation results
    in_est_res_filename = est_res_filename_pattern.format(in_est_res_number)
    with open(in_est_res_filename, "rb") as f:
        in_est_res = pickle.load(f)
    kernels_types = in_est_res["kernels_types"]
    estimation_params = in_est_res["estimation_params"]
    estimated_params = in_est_res["estimated_params"]
    selected_clusters = in_est_res["selected_clusters"]
    selected_trials_ids = in_est_res["selected_trials_ids"]

    leg_quad_points = estimation_params["ell_calculation_params"]["leg_quad_points"]
    leg_quad_weights = estimation_params["ell_calculation_params"]["leg_quad_weights"]

    # get estimation parameters
    variational_mean = estimated_params["variational_mean"]
    variational_chol_vecs = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    # subset selected_clusters
    spikes_times = striatumUtils.subset_clusters_data(
        selected_clusters=selected_clusters,
        clusters=clusters_ids,
        spikes_times=spikes_times,
    )

    # get selected_trials_ids
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
    n_latents = C.shape[1]

    est_init_filename = est_init_filename_pattern.format(est_init_number)
    est_init = configparser.ConfigParser()
    est_init.read(est_init_filename)

    optim_params = dict(
        # n_quad=int(est_init["optim_params"]["n_quad"]),
        jit=bool(est_init["optim_params"]["in_steps_jit"]),
        # maxiter=int(est_init["optim_params"]["in_steps_maxiter"]),
        maxiter=maxiter,
        tol=float(est_init["optim_params"]["in_steps_tol"]),
        max_stepsize=float(est_init["optim_params"]["in_steps_max_stepsize"]),
        history_size=int(est_init["optim_params"]["in_steps_history_size"]),
        em_tol=float(est_init["optim_params"]["in_steps_em_tol"]),
        max_cont_lb_below_thr=int(est_init["optim_params"]["in_steps_max_cont_lb_below_thr"]),
        n_updates_btw_reports=int(est_init["optim_params"]["n_updates_btw_reports"]),
    )

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    # save estimation initial conditions
    out_est_metadata = configparser.ConfigParser()
    out_est_metadata["script_info"] = {
        "name": __file__,
    }
    out_est_metadata["data_params"] = {
        "selected_trials_ids": selected_trials_ids,
        "selected_clusters": selected_clusters,
        "clusters_ids": clusters_ids,
        "nLatents": n_latents,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    out_est_metadata["estimation_params"] = {
        "est_init_number": est_init_number,
        "in_est_res_number": in_est_res_number,
    }

    # build model_save_filename
    estPrefixUsed = True
    while estPrefixUsed:
        out_est_res_number = random.randint(0, 10**8)
        out_est_metadata_filename = \
            metadata_filename_pattern.format(out_est_res_number)
        if not os.path.exists(out_est_metadata_filename):
            estPrefixUsed = False
    out_est_res_filename = est_res_filename_pattern.format(out_est_res_number)


    with open(out_est_metadata_filename, "w") as f:
        out_est_metadata.write(f)
    print(f"Saved {out_est_metadata_filename}")

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
        C=C,
        d=d,
        ind_points_locs=ind_points_locs,
        kernels_params=kernels_params,
    )
    # def optim_func(params, additional_params):
    def optim_func(params):
        value = em._eval_func(
            vMean=params["variational_mean"],
            vChol=params["variational_chol_vecs"],
            C=params["C"], d=params["d"],
            kernels_params=params["kernels_params"],
            ind_points_locs=params["ind_points_locs"],
        )
        return value

    start_time = time.time()
    # res = em.maximize_jaxopt_LBFGS_in_steps(optim_func=optim_func,
    #                                         params0=params0,
    #                                         optim_params=optim_params,
    #                                        )
    del optim_params["em_tol"]
    del optim_params["max_cont_lb_below_thr"]
    del optim_params["n_updates_btw_reports"]
    res = em.maximize_jaxopt_LBFGS_one_call(optim_func=optim_func,
                                            params0=params0,
                                            optim_params=optim_params,
                                           )
    elapsed_time = time.time() - start_time

    # out_est_res = res.copy()
    # out_est_res["estimated_params"] = out_est_res.pop("params")
    out_est_res = dict()
    out_est_res["estimated_params"] = res.params
    out_est_res["estimated_state"] = res.state
    out_est_res["fixed_params"] = None
    out_est_res["selected_trials_ids"] = selected_trials_ids
    out_est_res["selected_clusters"] = selected_clusters
    out_est_res["clusters_ids"] = clusters_ids
    out_est_res["kernels_types"] = kernels_types
    out_est_res["estimation_params"] = estimation_params
    out_est_res["optim_params"] = optim_params
    out_est_res["trials_start_times"] = trials_start_times
    out_est_res["trials_end_times"] = trials_end_times
    out_est_res["epochs_times"] = epochs_times

    with open(out_est_res_filename, "wb") as f:
        pickle.dump(out_est_res, f)
    print("Saved results to {:s}".format(out_est_res_filename))


if __name__ == "__main__":
    main(sys.argv)
