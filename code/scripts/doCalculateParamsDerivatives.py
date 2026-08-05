
import sys
import jax
import numpy as np
import pickle
import argparse
import configparser

import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def get_derivatives(loss_fn, params):
    derivatives_fn = jax.grad(loss_fn)
    derivatives = derivatives_fn(params)
    return derivatives


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
                        # default=33576128)
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=26)
    parser.add_argument("--est_init_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        default="../../metadata/trialsIDsFrom30100To30199.csv")
                        # default="../../metadata/trialsIDsFrom30200To30299.csv")
                        # default="../../metadata/trialsIDsFrom30000To30099.csv")
                        # default="../../metadata/trialsIDsFrom242To341.csv")
                        # default="../../metadata/trialsIDsFrom142To241.csv")
                        # default="../../metadata/trialsIDsFrom42To141.csv")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filename",
                        type=str,
                        default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/23323766_pseudo_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/42430740_shuffled_pseudo_epoched_spikes_times.pickle")
    parser.add_argument("--est_res_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--derivatives_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_derivatives.pickle")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    est_init_number = args.est_init_number
    est_init_filename_pattern = args.est_init_filename_pattern
    trials_ids_filename = args.trials_ids_filename
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    est_res_filename_pattern = args.est_res_filename_pattern
    derivatives_filename = args.derivatives_filename_pattern.format(
		est_res_number)

    # get spike_times
    with open(epoched_spikes_times_filename, "rb") as f:
        epoch_spikes_res = pickle.load(f)
    spikes_times = epoch_spikes_res["spikes_times"]
    trials_ids = epoch_spikes_res["trials_ids"]
    trials_start_times = np.array(epoch_spikes_res["trials_start_times"])
    trials_end_times = np.array(epoch_spikes_res["trials_end_times"])
    epochs_times = np.array(epoch_spikes_res["epochs_times"])

    est_res_filename = est_res_filename_pattern.format(est_res_number)
    with open(est_res_filename, "rb") as f:
        est_res = pickle.load(f)
    kernels_types = est_res["kernels_types"]
    estimation_params = est_res["estimation_params"]
    estimated_params = est_res["estimated_params"]
    selected_clusters = est_res["selected_clusters"]
    clusters_ids = est_res["clusters_ids"]

    variational_mean = estimated_params["variational_mean"]
    variational_chol_vecs = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    # kernels_params = estimated_params["kernels_params"]

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

    #    build dynamic parameter specifications
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params_spec = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #   build config file parameters specification
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init).get_dict()
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

    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]

    optim_params = dict(
        n_quad=int(est_init["optim_params"]["n_quad"]),
        jit=bool(est_init["optim_params"]["in_steps_jit"]),
        maxiter=int(est_init["optim_params"]["in_steps_maxiter"]),
        tol=float(est_init["optim_params"]["in_steps_tol"]),
        max_stepsize=float(est_init["optim_params"]["in_steps_max_stepsize"]),
        history_size=int(est_init["optim_params"]["in_steps_history_size"]),
        em_tol=float(est_init["optim_params"]["in_steps_em_tol"]),
        max_cont_lb_below_thr=int(est_init["optim_params"]["in_steps_max_cont_lb_below_thr"]),
    )

    leg_quad_points, leg_quad_weights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            n_quad=optim_params["n_quad"],
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)
    del optim_params["n_quad"]
    estimation_params["ell_calculation_params"]["leg_quad_points"] = \
        leg_quad_points
    estimation_params["ell_calculation_params"]["leg_quad_weights"] = \
        leg_quad_weights

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
        kernels_params=kernels_params0,
    )
    def inferenceOptimFunc(params):
        value = em._eval_func(
            vMean=params["variational_mean"],
            vChol=params["variational_chol_vecs"],
            C=params["C"], d=params["d"],
            kernels_params=params["kernels_params"],
            ind_points_locs=params["ind_points_locs"],
        )
        return value

    derivatives = get_derivatives(loss_fn=inferenceOptimFunc, params=params0)

    with open(derivatives_filename, "wb") as f: pickle.dump(derivatives, f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
