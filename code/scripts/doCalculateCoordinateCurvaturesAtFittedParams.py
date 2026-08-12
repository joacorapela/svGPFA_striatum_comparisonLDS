
import sys
import jax
import numpy as np
import pickle
import argparse
import configparser

import svGPFA.stats.em
import svGPFA.utils.miscUtils

import striatumUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--micro_batch_size", help="micro batch size",
                        type=int, default=5)
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
                        # default=33576128)
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
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.pickle")
    args = parser.parse_args()

    micro_batch_size = args.micro_batch_size
    est_res_number = args.est_res_number
    metadata_filename = args.metadata_filename_pattern.format(est_res_number)
    est_res_filename_pattern = args.est_res_filename_pattern
    curvatures_filename = args.curvatures_filename_pattern.format(
		est_res_number)

    # get spike_times
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    epoched_spikes_times_filename = metadata["data_params"]["epoched_spikes_times_filename"]
    est_init_number = int(metadata["estimation_params"]["est_init_number"])

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
    common_n_ind_points = variational_mean.shape[2]
    n_ind_points = [common_n_ind_points] * n_latents

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)

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
    params = dict(
        variational_mean=variational_mean,
        variational_chol_vecs=variational_chol_vecs,
        C=C,
        d=d,
        ind_points_locs=ind_points_locs,
        kernels_params=kernels_params,
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

    curvatures = jaxUtils.get_coordinated_curvatures(
        loss_fn=inferenceOptimFunc,
        params=params,
        micro_batch_size=micro_batch_size)

    with open(curvatures_filename, "wb") as f: pickle.dump(curvatures, f)

    print(f"Saved {curvatures_filename}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
