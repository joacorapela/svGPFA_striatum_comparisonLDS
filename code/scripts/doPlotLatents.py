
import sys
import warnings
import configparser
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle
import argparse
import plotly.colors

import svGPFA.utils.miscUtils
import svGPFA.utils.statsUtils
import svGPFA.plot.plotUtilsPlotly
import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        # default=69706576)
                        # default=38426992)
                        # default=99226606)
                        # default=99749566)
                        default=54368807)
                        # default=42976344)
                        # default=34655634)
                        # default=22746506)
                        # default=92418550)
                        # default=25058234)
                        # default=6092425)
                        # default=1283092)
                        # default=57339587)
                        # default=33576128)
                        # default=74463115)
                        # default=71504301)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
#     parser.add_argument("--filepath", help="dandi filepath", type=str,
#                         default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--latent_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--latents_to_2D_plot", help="latents to plot in 2D plot",
                        type=str, default="[0,1]")
    parser.add_argument("--latents_to_3D_plot", help="latents to plot in 3D plot",
                        type=str, default="[0,1,2]")
    parser.add_argument("--ports_to_plot",
                        help="ports to plot", type=str,
                        default="1,2,3,4,5,6,7")
    parser.add_argument("--ports_markers_str",
                        help="markers for ports", type=str,
                        default="circle,circle,circle,circle,circle,circle,circle")
    parser.add_argument("--ports_colors_str",
                        help="colors for ports", type=str,
                        default="green,red,cyan,yellow,purple,blue,magenta")
    parser.add_argument("--colorscale_name",
                        help="colorscale name", type=str,
                        default="Light24")
#     parser.add_argument("--cluster", help="cluster to plot",
#                         type=int, default=10)
#     parser.add_argument("--align_event_name",
#                         help="name of event used for alignment",
#                         type=str,
#                         default="move_onset_time")
#     parser.add_argument("--events_names",
#                         help="names of marked events (e.g., start_time, target_on_time, go_cue_time, move_onset_time, stop_time)",
#                         type=str,
#                         default="[start_time,target_on_time,go_cue_time,move_onset_time,stop_time]")
#     parser.add_argument("--events_colors",
#                         help="colors for marked events (e.g., start_time, target_on_time, go_cue_time, move_onset_time, stop_time)",
#                         type=str, default="[black,cyan,magenta,orange,pink]")
#     parser.add_argument("--events_markers",
#                         help="markers for marked events (e.g., start_time, target_on_time, go_cue_time, move_onset_time, stop_time)",
#                         type=str, default="[circle,circle,circle,circle,circle]")
    parser.add_argument("--model_filename_pattern",
                        help="model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_{:s}.pickle")
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_{:s}_metaData.ini")
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inference_metaData.ini")
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--rewarded_trials_times_filename", type=str,
                        help="rewarded trials times filename",
                        default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/collaborations/emmett/repo/results/EJT178_implant1/recording6_29-03-2022/trials_times.csv")
    parser.add_argument("--clustering_res_filename", type=str,
                        help="clustering result filename",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/80586545_trials_times_clustering_result.pickle")
    parser.add_argument("--latents_fig_filename_pattern",
                        help="latents figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_latent{:03d}.{{:s}}")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
    parser.add_argument("--ortonormalized_latent_fig_filename_pattern",
                        help="orthonormalized latent figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_orthonormalized_latent{:03d}.{{:s}}")
    parser.add_argument("--ortonormalized_latents_fig_filename_pattern",
                        help="orthonormalized latents figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_orthonormalized_latents{:s}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inferred = args.inferred
#     filepath = args.filepath
    latent_to_plot = args.latent_to_plot
    latents_to_3D_plot = [int(str) for str in args.latents_to_3D_plot[1:-1].split(",")]
    latents_to_2D_plot = [int(str) for str in args.latents_to_2D_plot[1:-1].split(",")]
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_markers_str = args.ports_markers_str.split(",")
    ports_colors_str = args.ports_colors_str.split(",")
    colorscale_name = args.colorscale_name
    if inferred:
        model_filename = args.model_filename_pattern.format(est_res_number,
                                                            "inferredModel")
    else:
        model_filename = args.model_filename_pattern.format(est_res_number,
                                                            "estimatedModel")
    if inferred:
        metadata_filename = args.metadata_filename_pattern.format(
            est_res_number, "inference")
    else:
        metadata_filename = args.metadata_filename_pattern.format(
            est_res_number, "estimation")
    rewarded_trials_times_filename = args.rewarded_trials_times_filename
    clustering_res_filename = args.clustering_res_filename
    latents_fig_filename_pattern = args.latents_fig_filename_pattern.format(est_res_number, latent_to_plot)
    transitions_data_filename = args.transitions_data_filename
    ortonormalized_latent_fig_filename_pattern = args.ortonormalized_latent_fig_filename_pattern.format(est_res_number, latent_to_plot)
    ortonormalized_latents_fig_filename_pattern = args.ortonormalized_latents_fig_filename_pattern

    latents_to_2D_plot_str = "".join(str(i)+"_" for i in latents_to_2D_plot)
    latents_to_3D_plot_str = "".join(str(i)+"_" for i in latents_to_3D_plot)
    orthonormalized_latents2D_fig_filename_pattern = ortonormalized_latents_fig_filename_pattern.format(est_res_number, latents_to_2D_plot_str)
    orthonormalized_latents3D_fig_filename_pattern = ortonormalized_latents_fig_filename_pattern.format(est_res_number, latents_to_3D_plot_str)

    transitions_data = pd.read_csv(transitions_data_filename)

#     cluster = args.cluster
#     align_event_name = args.align_event_name
#     events_names = [str for str in args.events_names[1:-1].split(",")]
#     events_colors = [str for str in args.events_colors[1:-1].split(",")]
#     events_markers = [str for str in args.events_markers[1:-1].split(",")]

#     with NWBHDF5IO(filepath, 'r') as io:
#         nwbfile = io.read()
#         trials_df = nwbfile.intervals["trials"].to_dataframe()

#     trials_colors_patterns = utils.get_trials_colors_patterns(trials_df=trials_df)
#     trials_colors = [trial_color_pattern.format(1.0)
#                      for trial_color_pattern in trials_colors_patterns]

    metaData = configparser.ConfigParser()
    metaData.read(metadata_filename)
    epoched_spikes_times_filename = metaData["data_params"]["epoched_spikes_times_filename"]

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    epochs_trials_ids = load_res["trials_ids"]
    epochs_times = load_res["epochs_times"]

    with open(model_filename, "rb") as f:
        est_results = pickle.load(f)
    final_lower_bound = est_results["lower_bound_hist"][-1]
    trials_ids = est_results["trials_ids"].tolist()
    kernels_types = est_results["kernels_types"]
    clusters = est_results["selected_clusters"]

    # clusters = est_results["clusters_ids"]
    # clusters = est_results["clusters"]
    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    # reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    reg_param = 1e-5
    estimated_params = est_results["estimated_params"]
    if inferred:
        fixed_params = est_results["fixed_params"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    kernels_params = estimated_params["kernels_params"]
    if inferred:
        C = fixed_params["C"]
        d = fixed_params["d"]
    else:
        C = estimated_params["C"]
        d = estimated_params["d"]
    ind_points_locs = estimated_params["ind_points_locs"]

    # extract latents means and varances and estimated C
    l_means, l_vars = svGPFA.utils.statsUtils.computeLatents(
        vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    times = jnp.asarray(leg_quad_points)

    l_means = np.transpose(jnp.asarray(l_means), (1, 2, 0))
    l_vars = np.transpose(jnp.asarray(l_vars), (1, 2, 0))
    estimatedC = jnp.asarray(C)

    # extract markers
    ports_markers = dict(zip(ports_to_plot, ports_markers_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    marked_events_times, marked_events_colors, marked_events_markers, \
            marked_events_labels = plotUtils.buildMarkedEventsInfoFromTransitions(
                transitions_data=transitions_data, trials_ids=trials_ids)

    align_event_times = []
    for r, trial_id in enumerate(trials_ids):
        epoch_trial_index = np.where(epochs_trials_ids == trial_id)[0][0]
        align_event_times.append(epochs_times[epoch_trial_index])

    events_names = ["trial {:d}".format(trial_id)
                    for trial_id in trials_ids]
                    # for trial_id in marked_events_trials_ids]

    # n_trials = len(marked_events_trials_ids)
    n_trials = len(trials_ids)
    colorscale = plotly.colors.qualitative.Light24
    colorscale_rgb = [plotly.colors.hex_to_rgb(hex_color)
                      for hex_color in colorscale]
    colors_patterns = ["rgba({:d},{:d},{:d},{{:f}})".format(color[0], color[1], color[2]) for color in colorscale_rgb]
    trials_colors_patterns = [colors_patterns[r%len(colors_patterns)] for r in range(n_trials)]
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    title = f"Lower bound: {final_lower_bound:.02f}"

    # plot estimated latent across trials
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
        times=times,
        latentsMeans=l_means,
        latentsSTDs=np.sqrt(l_vars),
        latentToPlot=latent_to_plot,
        trials_ids=trials_ids,
        align_event_times=align_event_times,
        events_names=events_names,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors_patterns=trials_colors_patterns,
        xlabel="Time (sec)")
    fig.update_layout(title=title)
    fig.write_image(latents_fig_filename_pattern.format("png"))
    fig.write_html(latents_fig_filename_pattern.format("html"))

    rewarded_trials_times = pd.read_csv(rewarded_trials_times_filename)
    with open(clustering_res_filename, "rb") as f:
        load_res = pickle.load(f)
    rs = load_res["rs"]
    means = load_res["means"]

    if means[0] < means[1]:
        short_trials_index = 0
    else:
        short_trials_index = 1
    short_trials_cluster_responsibilities = rs[:, short_trials_index]

    fig = plotUtils.getPlotOrthonormalizedLatentAcrossTrials(
        times=times, latentsMeans=l_means, latentsVars=l_vars,
        C=estimatedC, trials_ids=trials_ids,
        latentToPlot=latent_to_plot,
        align_event_times=align_event_times,
        rewarded_trials_times=rewarded_trials_times,
        short_trials_cluster_responsibilities=short_trials_cluster_responsibilities,
        events_names=events_names,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors_patterns=trials_colors_patterns,
#         trials_annotations=trials_annotations,
        xlabel="Time (sec)")
    fig.update_layout(title=title)
    fig.write_image(ortonormalized_latent_fig_filename_pattern.format("png"))
    fig.write_html(ortonormalized_latent_fig_filename_pattern.format("html"))

    breakpoint()

    fig = svGPFA.plot.plotUtilsPlotly.get2DPlotOrthonormalizedLatentsAcrossTrials(
        trials_times=times, latentsMeans=l_means, latentsVars=l_vars,
        C=estimatedC, trials_ids=trials_ids,
        latentsToPlot=latents_to_2D_plot,
        align_event_times=align_event_times,
        events_names=events_names,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors=trials_colors,
        # trials_annotations=trials_annotations,
    )
    fig.update_layout(title=title)
    fig.write_image(orthonormalized_latents2D_fig_filename_pattern.format("png"))
    fig.write_html(orthonormalized_latents2D_fig_filename_pattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
        trials_times=times, latentsMeans=l_means, latentsVars=l_vars,
        C=estimatedC, trials_ids=trials_ids,
        latentsToPlot=latents_to_3D_plot,
        align_event_times=align_event_times,
        events_names=events_names,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors=trials_colors,
        # trials_annotations=trials_annotations,
    )
    fig.update_layout(title=title)
    fig.write_image(orthonormalized_latents3D_fig_filename_pattern.format("png"))
    fig.write_html(orthonormalized_latents3D_fig_filename_pattern.format("html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
