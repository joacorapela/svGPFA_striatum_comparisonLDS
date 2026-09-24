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
    parser.add_argument("--in_est_res_number", help="input estimation result number",
                        type=int,
                        default=54368807)
                        # default=33576128)
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int,
                        default=41)
                        # default=34)
                        # default=36)
    parser.add_argument("--precondition", help="precondition estimation",
                        action="store_true")
    parser.add_argument("--est_init_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        default="../../metadata/trialsIDsFrom40100To40199.csv")
                        # default="../../metadata/trialsIDsFrom30100To30199.csv")
                        # default="../../metadata/trialsIDsFrom30200To30299.csv")
                        # default="../../metadata/trialsIDsFrom30000To30099.csv")
                        # default="../../metadata/trialsIDsFrom242To341.csv")
                        # default="../../metadata/trialsIDsFrom142To241.csv")
                        # default="../../metadata/trialsIDsFrom42To141.csv")
    parser.add_argument("--model_curvatures_filename_pattern",
                        help="model curvatures filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.pickle")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filename",
                        type=str,
                        default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/79695807_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/23323766_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/23323766_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/42430740_shuffled_pseudo_epoched_spikes_times.pickle")
    parser.add_argument("--est_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--est_res_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    args = parser.parse_args()

    in_est_res_number = args.in_est_res_number
    est_init_number = args.est_init_number
    precondition = args.precondition
    est_init_filename_pattern = args.est_init_filename_pattern
    trials_ids_filename = args.trials_ids_filename
    model_curvatures_filename = args.model_curvatures_filename_pattern.format(
        in_est_res_number)
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    est_metadata_filename_pattern = args.est_metadata_filename_pattern
    est_res_filename_pattern = args.est_res_filename_pattern

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        epoch_spikes_res = pickle.load(f)
    spikes_times = epoch_spikes_res["spikes_times"]
    trials_ids = epoch_spikes_res["trials_ids"]
    trials_start_times = np.array(epoch_spikes_res["trials_start_times"])
    trials_end_times = np.array(epoch_spikes_res["trials_end_times"])
    epochs_times = np.array(epoch_spikes_res["epochs_times"])

    in_est_res_filename = est_res_filename_pattern.format(in_est_res_number)
    with open(in_est_res_filename, "rb") as f:
        in_est_res = pickle.load(f)
    kernels_types = in_est_res["kernels_types"]
    estimation_params = in_est_res["estimation_params"]
    estimated_params = in_est_res["estimated_params"]
    selected_clusters = in_est_res["selected_clusters"]
    clusters_ids = in_est_res["clusters_ids"]

    variational_mean = estimated_params["variational_mean"]
    variational_chol_vecs = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]

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

    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    n_latents = C.shape[1]
    common_n_ind_points = variational_mean.shape[2]
    n_ind_points = [common_n_ind_points] * n_latents

    est_init_filename = est_init_filename_pattern.format(est_init_number)
    est_init = configparser.ConfigParser()
    est_init.read(est_init_filename)

    #   build config file parameters specification
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init).get_dict()
    config_file_params_spec = \
        svGPFA.utils.initUtils.getParamsDictFromStringsDict(
            n_latents=n_latents, n_trials=n_trials,
            strings_dict=strings_dict, args_info=args_info)
    kernels_params0, kernels_types = \
        svGPFA.utils.initUtils.getKernelsParams0AndTypes(
            n_latents=n_latents,
            config_file_params_spec=config_file_params_spec)

    optim_params = dict(
        jit=bool(est_init["optim_params"]["in_steps_jit"]),
        maxiter=int(est_init["optim_params"]["in_steps_maxiter"]),
        tol=float(est_init["optim_params"]["in_steps_tol"]),
        max_stepsize=float(est_init["optim_params"]["in_steps_max_stepsize"]),
        history_size=int(est_init["optim_params"]["in_steps_history_size"]),
        em_tol=float(est_init["optim_params"]["in_steps_em_tol"]),
        max_cont_lb_below_thr=int(est_init["optim_params"]["in_steps_max_cont_lb_below_thr"]),
        n_updates_btw_reports=int(est_init["optim_params"]["n_updates_btw_reports"]),
    )

    leg_quad_points, leg_quad_weights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            n_quad=int(est_init["optim_params"]["n_quad"]),
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    ind_points_locs = svGPFA.utils.initUtils.buildEquidistantIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        n_ind_points=n_ind_points,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    # save estimation initial conditions for provenance tracking
    out_est_metadata = configparser.ConfigParser()
    out_est_metadata["script_info"] = {
        "name": __file__,
    }
    out_est_metadata["data_params"] = {
        "trials_ids": selected_trials_ids,
        "selected_clusters": selected_clusters,
        "clusters_ids": clusters_ids,
        "nLatents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    out_est_metadata["estimation_params"] = {
        "in_est_res_number": in_est_res_number,
        "est_init_number": est_init_number,
        "precondition": precondition,
    }

    # build model_save_filename
    estPrefixUsed = True
    while estPrefixUsed:
        est_res_number = random.randint(0, 10**8)
        out_est_metadata_filename = \
            est_metadata_filename_pattern.format(est_res_number)
        if not os.path.exists(out_est_metadata_filename):
            estPrefixUsed = False
    out_est_res_filename = est_res_filename_pattern.format(est_res_number)


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
        ind_points_locs=ind_points_locs,
        kernels_params=kernels_params0,
    )
    additional_params = dict(
        C=C,
        d=d,
    )
    def optim_func(params, additional_params):
        value = em._eval_func(
            vMean=params["variational_mean"],
            vChol=params["variational_chol_vecs"],
            C=additional_params["C"],
            d=additional_params["d"],
            kernels_params=params["kernels_params"],
            ind_points_locs=params["ind_points_locs"],
        )
        return value

    if precondition:
        with open(model_curvatures_filename, "rb") as f:
            curvatures = pickle.load(f)

        preconditioning_epsilon = float(
            est_init["optim_params"]["preconditioning_epsilon"]
        )

        # Prevent extreme scaling by capping scale factors
        scale_raw = jax.tree_util.tree_map(
            lambda c: 1.0 / (jnp.sqrt(jnp.abs(c)) + preconditioning_epsilon),
            curvatures
        )
        del scale_raw["C"]
        del scale_raw["d"]

        # Clip scale factor
        preconditioning_min_scale_factor = \
            float(est_init["optim_params"]["preconditioning_min_scale_factor"])
        preconditioning_max_scale_factor = \
            float(est_init["optim_params"]["preconditioning_max_scale_factor"])
        params_scale = jax.tree_util.tree_map(
            lambda s: jnp.clip(s, a_min=preconditioning_min_scale_factor,
                               a_max=preconditioning_max_scale_factor),
            scale_raw
        )

        def preconditioned_optim_func(params, additional_params):
            # Unscale parameters back to original domain using element-wise multiplication
            unscaled_params = jax.tree_util.tree_map(
                lambda p, s: p * s, params, params_scale
            )
            answer = optim_func(params=unscaled_params,
                                additional_params=additional_params)
            return answer

        scaled_params0 = jax.tree_util.tree_map(
            lambda p, s: p / s, params0, params_scale
        )

        start_time = time.time()
        res = em.maximize_jaxopt_LBFGS_in_steps(optim_func=preconditioned_optim_func,
                                                params0=scaled_params0,
                                                additional_params=additional_params,
                                                optim_params=optim_params,
                                               )
        elapsed_time = time.time() - start_time

        unscaled_estimated_params = jax.tree_util.tree_map(
            lambda p, s: p * s, res["params"], params_scale
        )
        del res["params"]

    else:
        start_time = time.time()
        res = em.maximize_jaxopt_LBFGS_in_steps(optim_func=optim_func,
                                                params0=params0,
                                                additional_params=additional_params,
                                                optim_params=optim_params,
                                               )
        elapsed_time = time.time() - start_time

    # save the new leg_quad_points and leg_quad_weights
    estimation_params["ell_calculation_params"]["leg_quad_points"] = \
        leg_quad_points
    estimation_params["ell_calculation_params"]["leg_quad_weights"] = \
        leg_quad_weights

    # save the estimation results for provenance tracking
    out_est_res = res.copy()
    if precondition:
        out_est_res["estimated_params"] = unscaled_estimated_params
    else:
        out_est_res["estimated_params"] = out_est_res.pop("params")
    out_est_res["fixed_params"] = additional_params
    out_est_res["trials_ids"] = selected_trials_ids
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

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
