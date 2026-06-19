import sys
import pickle
import argparse
import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go

import svGPFA.utils.miscUtils
import svGPFA.utils.statsUtils
import plotUtils
import striatumUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
    parser.add_argument("--inf_res_number", help="estimation result number",
                        type=int,
                        default=84848570)
                        # default=7996538)
                        # default=91676545)
                        # default=71005668)
                        # default=87796368)
    parser.add_argument("--latents_sample_rate", help="plot sample rate",
                        type=int, default=50)
    parser.add_argument("--test_latent_trial", help="trial of the test latent",
                        type=int,
                        default=42)
    parser.add_argument("--latent_indices", help="indices of test and inferred latents",
                        type=str,
                        default="[0,1,2,3,4]")
    parser.add_argument("--estimated_model_filename_pattern",
                        help="saved estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--inferred_model_filename_pattern",
                        help="saved inferred model filename pattern", type=str,
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_{:08d}_kl_testLatentTrial_{:03d}_latentIndices{:s}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inf_res_number = args.inf_res_number
    latents_sample_rate = args.latents_sample_rate
    test_latent_trial = args.test_latent_trial
    latent_indices = [int(s) for s in args.latent_indices[1:-1].split(",")]
    est_model_filename_pattern = args.estimated_model_filename_pattern
    inf_model_filename_pattern = args.inferred_model_filename_pattern
    latent_indices_str = "_".join([str(i) for i in latent_indices])
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number,
        inf_res_number, test_latent_trial, latent_indices_str)

    est_model_filename = est_model_filename_pattern.format(est_res_number)
    inf_model_filename = inf_model_filename_pattern.format(inf_res_number)

    # get estimated params
    with open(est_model_filename, "rb") as f:
        est_results = pickle.load(f)
    trials_ids = est_results["trials_ids"].tolist()
    est_kernels_types = est_results["kernels_types"]
    est_leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    # reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    est_reg_param = 1e-5
    est_trials_ids = est_results["trials_ids"]
    estimated_params = est_results["estimated_params"]
    est_trials_start_times = est_results["trials_start_times"]
    est_trials_end_times = est_results["trials_end_times"]
    # est_epochs_times = est_results["epochs_times"]

    aux = np.where(est_trials_ids==test_latent_trial)
    if len(aux) == 0:
        raise ValueError(f"Trials {test_latent_trial} not found in {est_model_filename}")
    test_latent_trial_index = aux[0][0]

    est_vMean = estimated_params["variational_mean"]
    est_vChol = estimated_params["variational_chol_vecs"]
    est_C = estimated_params["C"]
    est_kernels_params = estimated_params["kernels_params"]
    est_ind_points_locs = estimated_params["ind_points_locs"]

    est_trials_times = svGPFA.utils.miscUtils.getEquispacedTrialsTimes(
        trials_start_times=est_trials_start_times,
        trials_end_times=est_trials_end_times,
        sample_rate=latents_sample_rate)

    est_l_means, est_l_vars = \
        svGPFA.utils.statsUtils.computeLatentsWithEquispacedTrialsTimes(
            vMean=est_vMean, vChol=est_vChol, kernels_params=est_kernels_params,
            ind_points_locs=est_ind_points_locs, kernels_types=est_kernels_types,
            trials_times=est_trials_times, reg_param=est_reg_param)

    est_l_means = [est_l_mean.T for est_l_mean in est_l_means]
    est_l_vars = [est_l_var.T for est_l_var in est_l_vars]
    est_C = jnp.asarray(est_C)
    est_ol_means, est_ol_vars = \
        svGPFA.utils.miscUtils.orthogonalizeLatentsWithEquispacedTrialsTimes(
            latents_means=est_l_means, latents_vars=est_l_vars, C=est_C)

    # get inferred params
    with open(inf_model_filename, "rb") as f:
        inf_results = pickle.load(f)
    trials_ids = inf_results["trials_ids"].tolist()
    inf_kernels_types = inf_results["kernels_types"]
    inf_leg_quad_points = inf_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    # reg_param = inf_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    inf_reg_param = 1e-5
    estimated_params = inf_results["estimated_params"]
    fixed_params = inf_results["fixed_params"]
    inf_trials_start_times = inf_results["trials_start_times"]
    inf_trials_end_times = inf_results["trials_end_times"]
    inf_epochs_times = inf_results["epochs_times"]

    inf_vMean = estimated_params["variational_mean"]
    inf_vChol = estimated_params["variational_chol_vecs"]
    inf_C = fixed_params["C"]
    inf_kernels_params = fixed_params["kernels_params"]
    inf_ind_points_locs = estimated_params["ind_points_locs"]

    inf_trials_times = svGPFA.utils.miscUtils.getEquispacedTrialsTimes(
        trials_start_times=inf_trials_start_times,
        trials_end_times=inf_trials_end_times,
        sample_rate=latents_sample_rate)

    inf_l_means, inf_l_vars = \
        svGPFA.utils.statsUtils.computeLatentsWithEquispacedTrialsTimes(
            vMean=inf_vMean, vChol=inf_vChol, kernels_params=inf_kernels_params,
            ind_points_locs=inf_ind_points_locs, kernels_types=inf_kernels_types,
            trials_times=inf_trials_times, reg_param=inf_reg_param)

    inf_l_means = [inf_l_mean.T for inf_l_mean in inf_l_means]
    inf_l_vars = [inf_l_var.T for inf_l_var in inf_l_vars]
    inf_C = jnp.asarray(inf_C)
    inf_ol_means, inf_ol_vars = \
        svGPFA.utils.miscUtils.orthogonalizeLatentsWithEquispacedTrialsTimes(
            latents_means=inf_l_means, latents_vars=inf_l_vars, C=inf_C)

    inf_times = jnp.asarray(inf_leg_quad_points)
    inf_times_non_epoched = striatumUtils.get_times_non_epoched(
        times=np.array(inf_times), epochs_times=np.array(inf_epochs_times))
    inf_ol_means_non_epoched = \
        striatumUtils.get_equispaced_latents_non_epoched(latents=inf_ol_means)
    inf_ol_vars_non_epoched = \
        striatumUtils.get_equispaced_latents_non_epoched(latents=inf_ol_vars)

    # computer KL divergences
    run_kl = [None] * len(latent_indices)
    run_kl_times = [None] * len(latent_indices)
    for i, latent_index in enumerate(latent_indices):
        test_pattern_mean = est_ol_means[test_latent_trial_index][:, latent_index]
        test_pattern_var = est_ol_vars[test_latent_trial_index][:, latent_index]
        time_series_mean = inf_ol_means_non_epoched[:, latent_index]
        time_series_var = inf_ol_vars_non_epoched[:, latent_index]
        run_kl[i] = striatumUtils.running_KL(
            test_pattern_mean=test_pattern_mean,
            test_pattern_var=test_pattern_var,
            time_series_mean=time_series_mean,
            time_series_var=time_series_var,
        )
        run_kl_times[i] = inf_times_non_epoched[:len(run_kl[i])]

    # plot results
    fig = go.Figure()
    for i in range(len(run_kl)):
        trace = go.Scatter(x=run_kl_times[i], y=run_kl[i], name=f"latent {latent_indices[i]}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="KL Divergence")
    fig.update_layout(title=f"Trial {test_latent_trial}")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print(f'Figure saved to {fig_filename_pattern.format("html")}')
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
