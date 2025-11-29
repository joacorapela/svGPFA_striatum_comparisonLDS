import sys
import os.path
import random
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import configparser
import cProfile

import gcnu_common.utils.neuralDataAnalysis
import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_init_number", help="estimation init number",
                        type=int)
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=10)
    parser.add_argument("--n_threads", help="number of threads for PyTorch",
                        type=int, default=6)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=15)
    parser.add_argument("--profile",
                        help="use this option if you want to profile svGPFA.maximize()",
                        action="store_true")
    parser.add_argument("--epoched_spikes_times_filename",
                        help="epoched spikes times filenamepattern",
                        type=str,
                        default="../../../svGPFA_striatum/results/spikes_times_epochedFirst2In_fixedDurationFalse_simplified.pickle")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str,
                        default="../../metadata/trialsFrom104To120.csv")
    parser.add_argument("--clusters_ids_filename", help="clusters ids filename",
                        type=str,
                        default="~/gatsby-swc/gatsby/code/ssm_rapela/repos/projects/lds_neuralLatents_striatum/metadata/clustersIndices_124_223.ini")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../metadata/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--profiling_info_filename_pattern",
                        help="profiling information filename pattern",
                        type=str,
                        default="../../results/{:08d}_profiling_info.txt")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_init_number = args.est_init_number
    n_latents = args.n_latents
    n_threads = args.n_threads
    common_n_ind_points = args.common_n_ind_points
    profile = args.profile
    epoched_spikes_times_filename = args.epoched_spikes_times_filename
    trials_ids_filename = args.trials_ids_filename
    clusters_ids_filename = args.clusters_ids_filename
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern
    profiling_info_filename_pattern = args.profiling_info_filename_pattern
    trials_ids_filename = args.trials_ids_filename
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
    regions = load_res["regions"]

    # breakpoint()

    # get selected_trials_ids
    selected_trials_ids = np.genfromtxt(trials_ids_filename, dtype=np.uint64)
    spikes_times, trials_start_times, trials_end_times = \
            striatumUtils.subset_trials_ids_data(
                selected_trials_ids=selected_trials_ids,
                trials_ids=trials_ids,
                spikes_times=spikes_times,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)

    # breakpoint()

    # get selected_clusters_ids
    selected_clusters_ids = np.genfromtxt(clusters_ids_filename, dtype=np.uint64)
    n_clusters = len(regions)
    clusters_indices = np.arange(n_clusters)
    units_to_remove = [n for n in range(n_clusters)
                       if regions[n] not in selected_regions]
    clusters_indices = np.delete(clusters_indices, units_to_remove)
    spikes_times = gcnu_common.utils.neuralDataAnalysis.removeUnits(
        spikes_times = spikes_times,
        units_to_remove=units_to_remove)

    # breakpoint()

    n_trials = len(spikes_times)
    # trials_indices = np.arange(n_trials)
    n_clusters = len(spikes_times[0])

    trials_durations = trials_end_times - trials_start_times
    spikes_times, clusters_indices = \
        gcnu_common.utils.neuralDataAnalysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
            spikes_times=spikes_times, clusters_indices=clusters_indices,
            trials_durations = trials_durations,
            min_neuron_trials_avg_firing_rate=min_neuron_trials_avg_firing_rate)
    clusters_ids = [clusters_ids[i] for i in clusters_indices]
    regions = [regions[i] for i in clusters_indices]

    # breakpoint()

    # spikes_times, trials_indices = \
    #     gcnu_common.utils.neuralDataAnalysis.removeTrialsLongerThanThr(
    #         spikes_times=spikes_times, trials_indices=trials_indices,
    #         trials_durations=trials_durations,
    #         max_trial_duration=max_trial_duration)

    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    # trials_ids = trials_ids[trials_indices]
    # trials_start_times = trials_start_times[trials_indices]
    # trials_end_times = trials_end_times[trials_indices]

    # breakpoint()

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
        n_clusters=n_clusters, n_trials=n_trials, n_latents=n_latents,
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
    if profile:
        profiling_info_filename_pattern = \
            profiling_info_filename_pattern.format(estResNumber)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # create model
    kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
    model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
        linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setParamsAndData(
        measurements=spikes_times,
        initial_params=params["initial_params"],
        eLLCalculationParams=params["ell_calculation_params"],
        priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

    # save estimated values
    estim_res_config = configparser.ConfigParser()
    estim_res_config["data_params"] = {
        "n_threads": n_threads,
        "selected_regions ": selected_regions,
        "trials_ids": selected_trials_ids,
        "clusters_indices": clusters_indices,
        "clusters_ids": clusters_ids,
        "regions": regions,
        "nLatents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        # "max_trial_duration": max_trial_duration,
        "min_neuron_trials_avg_firing_rate": min_neuron_trials_avg_firing_rate,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    estim_res_config["optim_params"] = params["optim_params"]
    estim_res_config["estimation_params"] = {"est_init_number":
                                             est_init_number}
    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)
    print(f"Saved {estim_res_metadata_filename}")

    # maximize lower bound
    def getSVPosteriorOnIndPointsParams(model, get_mean=True, latent=0, trial=0):
        params = model.getSVPosteriorOnIndPointsParams()
        base_index = 0
        if not get_mean:
            base_index = len(params)/2 - 1
        answer = params[base_index][trial, :, 0]
        return answer

    def getKernelsParams(model):
        params = model.getKernelsParams()
        return params

    # maximize lower bound
    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    if profile:
        pr = cProfile.Profile()
        pr.enable()

#     svGPFA.utils.my_globals.raise_exception = True

    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optim_params=params["optim_params"],
                      method=params["optim_params"]["optim_method"],
                      # getIterationModelParamsFn=getSVPosteriorOnIndPointsParams,
                      getIterationModelParamsFn=getKernelsParams,
                      printIterationModelParams=True)

    if profile:
        pr.disable()
        pr.dump_stats(filename=profiling_info_filename)

    resultsToSave = {"clusters_indices": clusters_indices,
                     "clusters_ids": clusters_ids,
                     "regions": regions,
                     "lowerBoundHist": lowerBoundHist,
                     "elapsedTimeHist": elapsedTimeHist,
                     "terminationInfo": terminationInfo,
                     "iterationModelParams": iterationsModelParams,
                     "spikes_times": spikes_times,
                     # "trials_indices": trials_indices,
                     "trials_ids": selected_trials_ids,
                     "trials_start_times": trials_start_times,
                     "trials_end_times": trials_end_times,
                     "epochs_times": epochs_times,
                     "model": model,
                    }
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
        print("Saved results to {:s}".format(modelSaveFilename))

    print(f"Elapsed time {elapsedTimeHist[-1]}")
    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
