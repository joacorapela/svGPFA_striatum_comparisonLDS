import sys
import os.path
import random
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import configparser

import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils
import jaxUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int,
                        default=17)
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=10)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=15)
    parser.add_argument("--micro_batch_size", help="micro batch size",
                        type=int, default=5)
    parser.add_argument("--preconditioning_epsilon",
                        help="preconditioning epsilon",
                        type=float,
                        default=1e-8)
    parser.add_argument("--curvatures_for_preconditioning_filename",
                        help="curvatures for preconditioning filename",
                        type=str,
                        default="")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filenamepattern",
                        type=str,
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/47205531_pseudo_epoched_spikes_times.pickle")
                        default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/96439322_epoched_spikes_times.pickle")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        # default="../../metadata/trialsIDsFrom20200To20299.csv")
                        # default="../../metadata/trialsIDsFrom42To141.csv")
                        # default="../../metadata/trialsIDsFrom142To241.csv")
                        # default="../../metadata/trialsIDs_372_414.csv")
                        default="../../metadata/trialsIDsFrom42To141.csv")
    parser.add_argument("--clusters_ids_filename", help="clusters ids filename",
                        type=str,
                        default="../../metadata/clustersIDs_striatum.csv")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--curvatures_filename_pattern",
                        help="curvatures filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.{:s}")
    parsed, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_init_number = args.est_init_number
    n_latents = args.n_latents
    common_n_ind_points = args.common_n_ind_points
    micro_batch_size = args.micro_batch_size
    preconditioning_epsilon = args.preconditioning_epsilon
    curvatures_for_preconditioning_filename = args.curvatures_for_preconditioning_filename
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    trials_ids_filename = args.trials_ids_filename
    clusters_ids_filename = args.clusters_ids_filename
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    curvatures_filename_pattern = args.curvatures_filename_pattern

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

    leg_quad_weights = params["ell_calculation_params"]["leg_quad_weights"]
    leg_quad_points = params["ell_calculation_params"]["leg_quad_points"]
    qMu0 = params["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["mean"].squeeze()
    variational_chol_vecs = params["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["cholVecs"]
    C = params["initial_params"]["embedding"]["C0"]
    d = params["initial_params"]["embedding"]["d0"]
    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
    Z0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["inducing_points_locs0"]

    # get estimation parameters
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
            # config_file_params_spec=config_file_params_spec)
            config_file_params_spec=config_file_params_spec,
            default_params_spec=default_params_spec)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # build spikes_times_array
    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=spikes_times)

    # build curvatures_filename
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
        "est_init_number": est_init_number,
        "n_latents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        "preconditioning_epsilon": preconditioning_epsilon,
        "curvatures_for_preconditioning_filename": curvatures_for_preconditioning_filename,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
        "trials_ids_filename": trials_ids_filename,
        "clusters_ids_filename": clusters_ids_filename,
        "est_init_config_filename_pattern": est_init_config_filename_pattern,
    }
    with open(curvatures_metadata_filename, "w") as f:
        curvatures_metadata.write(f)
    print(f"Saved {curvatures_metadata_filename}")

    # initialise loss function
    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=leg_quad_points, legQuadWeights=leg_quad_weights,
            reg_param=params["optim_params"]["prior_cov_reg_param"],
           )

    # define parameters
    params0 = dict(
        variational_mean=qMu0,
        variational_chol_vecs=variational_chol_vecs,
        C=C,
        d=d,
        kernels_params=kernels_params0,
        ind_points_locs=Z0,
    )

    # precondition parameters, if curvatures_for_preconditioning_filename is given
    if len(curvatures_for_preconditioning_filename) > 0:
        with open(curvatures_for_preconditioning_filename, "rb") as f:
            curvatures = pickle.load(curvatures_for_preconditioning_filename)
        params0["variational_mean"] /= jnp.sqrt(jnp.abs(curvatures["variational_mean"])) + preconditioning_epsilon
        params0["variational_chol_vecs"] /= jnp.sqrt(jnp.abs(curvatures["variational_chol_vecs"])) + preconditioning_epsilon
        params0["C"] /= jnp.sqrt(jnp.abs(curvatures["C"])) + preconditioning_epsilon
        params0["d"] /= jnp.sqrt(jnp.abs(curvatures["d"])) + preconditioning_epsilon
        params0["kernels_params"] = [kernel_param/(jnp.sqrt(jnp.abs(curvatures["kernels_params"][i]))+preconditioning_epsilon)
                                     for i, kernel_param in enumerate(params0["kernels_params"])]
        params0["ind_points_locs"] /= jnp.sqrt(jnp.abs(curvatures["ind_points_locs"])) + preconditioning_epsilon

    # define loss function
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

    # get curvatures
    curvatures = jaxUtils.get_coordinate_curvatures(
        loss_fn=loss_fn,
        params=params0,
        micro_batch_size=micro_batch_size)

    # save curvatures
    with open(curvatures_results_filename, "wb") as f:
        pickle.dump(curvatures, f)

    print(f"Saved {curvatures_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
