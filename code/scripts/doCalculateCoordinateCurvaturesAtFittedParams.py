
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

import svGPFA.stats.em
import svGPFA.utils.miscUtils

import striatumUtils
import jaxUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--micro_batch_size", help="micro batch size",
                        type=int, default=7)
    parser.add_argument("--est_res_number", help="model estimation result number",
                        type=int,
                        default=79151056)
                        # default=54368807)
                        # default=33576128)
    parser.add_argument("--precondition", help="precondition estimation",
                        action="store_true")
    parser.add_argument("--preconditioning_curvatures_number",
                        help="number of curvatures used for preconditioning",
                        type=int, default=7091558)
    parser.add_argument("--preconditioning_epsilon",
                        help=("small epsilon to add to every parameters "
                              "(to avoid division by zero)"),
                        type=float, default=1e-6)
    parser.add_argument("--preconditioning_min_scale_factor",
                        help="minimum scale factor for preconditioning",
                        type=float, default=1e-4)
    parser.add_argument("--preconditioning_max_scale_factor",
                        help="maximum scale factor for preconditioning",
                        type=float, default=1e+4)
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filenamepattern",
                        type=str,
                        default="")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/23323766_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/96439322_epoched_spikes_times.pickle")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        # default="../../metadata/trialsIDsFrom42To46.csv")
                        default="../../metadata/trialsIDsFrom30100To30199.csv")
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--est_res_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--curvatures_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.{:s}")
    args = parser.parse_args()

    micro_batch_size = args.micro_batch_size
    est_res_number = args.est_res_number
    precondition = args.precondition
    preconditioning_curvatures_number = args.preconditioning_curvatures_number
    preconditioning_epsilon = args.preconditioning_epsilon
    preconditioning_min_scale_factor = args.preconditioning_min_scale_factor
    preconditioning_max_scale_factor = args.preconditioning_max_scale_factor
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    trials_ids_filename = args.trials_ids_filename
    metadata_filename = args.metadata_filename_pattern.format(est_res_number)
    est_res_filename = args.est_res_filename_pattern.format(est_res_number)
    curvatures_filename_pattern = args.curvatures_filename_pattern

    preconditioning_curvatures_filename = curvatures_filename_pattern.format(
        preconditioning_curvatures_number, "pickle")

    # get estimation params
    with open(est_res_filename, "rb") as f:
        est_res = pickle.load(f)
    kernels_types = est_res["kernels_types"]
    estimation_params = est_res["estimation_params"]
    if "fixed_params" in est_res:
        fixed_params = est_res["fixed_params"]
    estimated_params = est_res["estimated_params"]
    selected_clusters = est_res["selected_clusters"]
    clusters_ids = est_res["clusters_ids"]
    if trials_ids_filename: # get selected_trials_ids from file
        selected_trials_ids = np.genfromtxt(trials_ids_filename, delimiter=',', dtype=np.uint64)
    else:                   # get trials ids from est_res
        if "selected_trials_ids" in est_res:
            selected_trials_ids = est_res["selected_trials_ids"]
        elif "trials_ids" in est_res:
            selected_trials_ids = est_res["trials_ids"]
        else:
            raise RuntimeError("selected_trials_ids or trials_ids should be in est_res")

    leg_quad_points = estimation_params["ell_calculation_params"]["leg_quad_points"]
    leg_quad_weights = estimation_params["ell_calculation_params"]["leg_quad_weights"]

    variational_mean = estimated_params["variational_mean"]
    variational_chol_vecs = estimated_params["variational_chol_vecs"]
    kernels_params = estimated_params["kernels_params"]
    if {"C", "d"}.issubset(set(estimated_params.keys())):
        C = estimated_params["C"]
        d = estimated_params["d"]
    elif {"C", "d"}.issubset(set(fixed_params.keys())):
        C = fixed_params["C"]
        d = fixed_params["d"]
    else:
        raise RuntimeError("C and d not found in estimated_params or fixed_params")
    ind_points_locs = estimated_params["ind_points_locs"]

    # get spike_times
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    if not epoched_spikes_times_filename:
        epoched_spikes_times_filename = metadata["data_params"]["epoched_spikes_times_filename"]
    est_init_number = int(metadata["estimation_params"]["est_init_number"])

    with open(epoched_spikes_times_filename, "rb") as f:
        epoch_spikes_res = pickle.load(f)
    spikes_times = epoch_spikes_res["spikes_times"]
    trials_ids = epoch_spikes_res["trials_ids"]
    trials_start_times = np.array(epoch_spikes_res["trials_start_times"])
    trials_end_times = np.array(epoch_spikes_res["trials_end_times"])
    epochs_times = np.array(epoch_spikes_res["epochs_times"])

    #   subset selected_clusters
    spikes_times = striatumUtils.subset_clusters_data(
        selected_clusters=selected_clusters,
        clusters=clusters_ids,
        spikes_times=spikes_times,
    )

    #   get selected_trials_ids
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

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    # build curvature filenames
    prefix_used = True
    while prefix_used:
        res_number = random.randint(0, 10**8)
        curvatures_metadata_filename = \
            curvatures_filename_pattern.format(res_number, "metadata")
        if not os.path.exists(curvatures_metadata_filename):
            prefix_used = False
    curvatures_results_filename = curvatures_filename_pattern.format(res_number, "pickle")

    # save cuvatures metdata
    curvatures_metadata = configparser.ConfigParser()
    curvatures_metadata["script_info"] = {
        "name": __file__,
    }
    curvatures_metadata["params"] = {
        "micro_batch_size": micro_batch_size,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
        "est_res_filename": est_res_filename,
        "precondition": precondition,
        "preconditioning_curvatures_filename": preconditioning_curvatures_filename,
        "preconditioning_epsilon": preconditioning_epsilon,
        "preconditioning_min_scale_factor": preconditioning_min_scale_factor,
        "preconditioning_max_scale_factor": preconditioning_max_scale_factor,
    }
    with open(curvatures_metadata_filename, "w") as f:
        curvatures_metadata.write(f)
    print(f"Saved {curvatures_metadata_filename}")

    # initialise loss_fn
    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=leg_quad_points, legQuadWeights=leg_quad_weights,
            reg_param=1e-5)
            # reg_param=optim_params["prior_cov_reg_param"])

    # calculate curvatures
    params = dict(
        variational_mean=variational_mean,
        variational_chol_vecs=variational_chol_vecs,
        C=C,
        d=d,
        ind_points_locs=ind_points_locs,
        kernels_params=kernels_params,
    )

    def loss_fn(params):
        value = em._eval_func(
            vMean=params["variational_mean"],
            vChol=params["variational_chol_vecs"],
            C=params["C"],
            d=params["d"],
            kernels_params=params["kernels_params"],
            ind_points_locs=params["ind_points_locs"],
        )
        return value

    if precondition:
        with open(preconditioning_curvatures_filename, "rb") as f:
            curvatures = pickle.load(f)

        # Prevent extreme scaling by capping scale factors
        scale_raw = jax.tree_util.tree_map(
            lambda c: 1.0 / (jnp.sqrt(jnp.abs(c)) + preconditioning_epsilon),
            curvatures
        )

        # Clip scale factor
        params_scale = jax.tree_util.tree_map(
            lambda s: jnp.clip(s, a_min=preconditioning_min_scale_factor,
                               a_max=preconditioning_max_scale_factor),
            scale_raw
        )

        def preconditioned_loss_fn(params):
            # Unscale parameters back to original domain using element-wise multiplication
            unscaled_params = jax.tree_util.tree_map(
                lambda p, s: p * s, params, params_scale
            )
            answer = loss_fn(params=unscaled_params)
            return answer

        scaled_params = jax.tree_util.tree_map(
            lambda p, s: p / s, params, params_scale
        )

        start_time = time.time()
        curvatures = jaxUtils.get_coordinate_curvatures(
            loss_fn=preconditioned_loss_fn,
            params=scaled_params,
            micro_batch_size=micro_batch_size)
        elapsed_time = time.time() - start_time

    else:
        start_time = time.time()
        curvatures = jaxUtils.get_coordinate_curvatures(
            loss_fn=loss_fn,
            params=params,
            micro_batch_size=micro_batch_size)
        elapsed_time = time.time() - start_time

    print(f"Elapsed time {elapsed_time}")

    with open(curvatures_results_filename, "wb") as f: pickle.dump(curvatures, f)

    print(f"Saved {curvatures_results_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
