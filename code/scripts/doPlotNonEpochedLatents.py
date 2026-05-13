
import sys
import warnings
import configparser
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle
import argparse
import plotly.graph_objects as go

import svGPFA.utils.miscUtils
import svGPFA.utils.statsUtils
import svGPFA.plot.plotUtilsPlotly
import plotUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=96281561)
                        # default=7996538)
                        # default=71005668)
                        # default=87796368)
                        # default=65877289)
                        # default=90996527)
                        # default=12870644)
                        # default=13338803)
                        # default=556223)
                        # default=23048734)
                        # default=93945415)
                        # default=72528369)
                        # default=13338803)
                        # default=556223)
                        # default=23048734)
                        # default=93945415)
                        # default=72528369)
                        # default=99226606)
                        # default=42833278)
                        # default=38426992)
                        # default=99226606)
                        # default=42976344)
                        # default=34655634)
                        # default=22746506)
                        # default=92418550)
                        # default=25058234)
                        # default=28563541)
                        # default=6092425)
                        # default=1283092)
                        # default=54368807)
                        # default=57339587)
                        # default=33576128)
                        # default=30172705)
                        # default=61182613)
                        # default=96671330)
                        # default=60672580)
                        # default=92037639)
                        # default=43638132)
                        # default=65922524)
                        # default=74463115)
                        # default=28719499)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--ports_to_plot",
                        help="ports to plot", type=str,
                        default="1,2,3,4,5,6,7")
    parser.add_argument("--ports_markers_str",
                        help="markers for ports", type=str,
                        default="circle,circle,circle,circle,circle,circle,circle")
    parser.add_argument("--in_ports_linetypes_str",
                        help="linetypes for in ports", type=str,
                        default="solid,solid,solid,solid,solid,solid,solid")
    parser.add_argument("--out_ports_linetypes_str",
                        help="linetypes for out ports", type=str,
                        default="dot,dot,dot,dot,dot,dot,dot")
    parser.add_argument("--ports_colors_str",
                        help="colors for ports", type=str,
                        default="green,red,cyan,yellow,purple,blue,magenta")
    parser.add_argument("--trials_boundaries_linetypes_str",
                        help="linetypes for trials boundaries", type=str,
                        default="solid,dash")
    parser.add_argument("--trials_boundaries_colors_str",
                        help="colors for trials boundaries", type=str,
                        default="black,black")
    parser.add_argument("--transition_data_filename",
                        help="transition data filename", type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
                        # default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/gatsby/svGPFA/repos/projects/svGPFA_striatum/data/Transition_data_sync.csv")
    parser.add_argument("--estimated_model_filename_pattern",
                        help="saved estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--inferred_model_filename_pattern",
                        help="saved inferred model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
    parser.add_argument("--latents_fig_filename_pattern",
                        help="latents figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_orthonormalized_nonEpoched_latents.{{:s}}")
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
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inferred = args.inferred
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_markers_str = args.ports_markers_str.split(",")
    in_ports_linetypes_str = args.in_ports_linetypes_str.split(",")
    out_ports_linetypes_str = args.out_ports_linetypes_str.split(",")
    ports_colors_str = args.ports_colors_str.split(",")
    trials_boundaries_linetypes_str = args.trials_boundaries_linetypes_str.split(",")
    trials_boundaries_colors_str = args.trials_boundaries_colors_str.split(",")
    transition_data_filename = args.transition_data_filename
    estimated_model_filename_pattern = args.estimated_model_filename_pattern
    inferred_model_filename_pattern = args.inferred_model_filename_pattern
    latents_fig_filename_pattern = args.latents_fig_filename_pattern.format(est_res_number)

    ports_markers = dict(zip(ports_to_plot, ports_markers_str))
    in_ports_linetypes = dict(zip(ports_to_plot, in_ports_linetypes_str))
    out_ports_linetypes = dict(zip(ports_to_plot, out_ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    if inferred:
        model_filename = inferred_model_filename_pattern.format(est_res_number)
    else:
        model_filename = estimated_model_filename_pattern.format(est_res_number)

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
        # fixed_params = est_results["fixed_params"][0]
    trials_start_times = est_results["trials_start_times"]
    trials_end_times = est_results["trials_end_times"]
    epochs_times = est_results["epochs_times"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    if inferred:
        C = fixed_params["C"]
        d = fixed_params["d"]
        kernels_params = fixed_params["kernels_params"]
    else:
        C = estimated_params["C"]
        d = estimated_params["d"]
        kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    # index to the original spikes
#     cluster_index = np.where(clusters==cluster)[0][0]

#     n_trials = len(spikes_times)
#     clusters_ids_str = " ".join(str(i) for i in clusters_ids)
#     if len(cluster_index) == 0:
#         raise ValueError("Cluster id {:d} is not valid. Valid cluster id are ".format(
#             cluster_index.item()) + clusters_ids_str)
# 
#     trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
#         start_times=trials_start_times,
#         end_times=trials_end_times,
#         n_steps=n_time_steps_CIF)
# 
#     trials_labels = np.array([str(i) for i in trials_ids])

#     n_trials = len(spikes_times)

#     trials_choices = [trials_info["choice"][trial_id]
#                       for trial_id in trials_ids]
#     trials_rewarded = [trials_info["feedbackType"][trial_id]
#                        for trial_id in trials_ids]
#     trials_contrast = [trials_info["contrastRight"][trial_id]
#                        if not np.isnan(trials_info["contrastRight"][trial_id])
#                        else trials_info["contrastLeft"][trial_id]
#                        for trial_id in trials_ids]
#     trials_colors_patterns = [choices_colors_patterns[0]
#                               if trials_choices[r] == -1
#                               else choices_colors_patterns[1]
#                               for r in range(n_trials)]
#     trials_colors = [trial_color_pattern.format(1.0)
#                      for trial_color_pattern in trials_colors_patterns]
#     trials_annotations = {"choice": trials_choices,
#                           "rewarded": trials_rewarded,
#                           "contrast": trials_contrast,
#                           "choice_prev": np.insert(trials_choices[:-1], 0,
#                                                    np.NAN),
#                           "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
#                                                      np.NAN)}
#

#     events_times = []
#     for event_name in events_names:
#         events_times.append([trials_df.iloc[trial_id][event_name]
#                              for trial_id in trials_ids])

#     marked_events_times, marked_events_colors, marked_events_markers = \
#         utils.buildMarkedEventsInfo(events_times=events_times,
#                                     events_colors=events_colors,
#                                     events_markers=events_markers)

#     align_event_times = [trials_df.iloc[trial_id][align_event_name]
#                          for trial_id in trials_ids]
    l_means, l_vars = svGPFA.utils.statsUtils.computeLatents(
        vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    l_means = np.transpose(jnp.asarray(l_means), (1, 2, 0))
    l_vars = np.transpose(jnp.asarray(l_vars), (1, 2, 0))
    estimatedC, _ = jnp.asarray(C), jnp.asarray(d)
    ol_means, ol_vars = svGPFA.utils.miscUtils.orthogonalizeLatents(
        latents_means=l_means, latents_vars=l_vars, C=estimatedC)

    times = jnp.asarray(leg_quad_points)
    times_non_epoched = plotUtils.get_times_non_epoched(
        times=np.array(times), epochs_times=np.array(epochs_times))
    ol_means_non_epoched = plotUtils.get_latents_non_epoched(latents=ol_means)
    ol_vars_non_epoched = plotUtils.get_latents_non_epoched(latents=ol_vars)

    n_latents = ol_means_non_epoched.shape[-1]
    latents_colors_patterns = plotUtils.getLatentsColorPatterns(
        n_latents=n_latents)

    start_time_sec = times_non_epoched.min()
    end_time_sec = times_non_epoched.max()

    transition_data = pd.read_csv(transition_data_filename)

    fig = plotUtils.getPlotNonEpochedLatents(
        times=times_non_epoched,
        latents_means=ol_means_non_epoched,
        latents_stds=np.sqrt(ol_vars_non_epoched),
#         events_names=events_names,
#         marked_events_times=marked_events_times,
#         marked_events_colors=marked_events_colors,
#         marked_events_markers=marked_events_markers,
#         trials_colors_patterns=trials_colors_patterns,
        latents_colors_patterns=latents_colors_patterns,
        xlabel="Time (sec)")

    ports_events_df = plotUtils.build_events_df(
        # trials_ids=trials_ids,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        transition_data=transition_data,
        in_ports_linetypes=in_ports_linetypes,
        out_ports_linetypes=out_ports_linetypes,
        # ports_markers=ports_markers,
        ports_colors=ports_colors)

    plotUtils.add_events_vlines(fig=fig, events_df=ports_events_df)

    bound_times = np.concatenate((trials_start_times+epochs_times,
                                  trials_end_times+epochs_times))
    bound_linetypes = ([trials_boundaries_linetypes_str[0]]*len(trials_start_times)+
                       [trials_boundaries_linetypes_str[1]]*len(trials_end_times))
    bound_colors = ([trials_boundaries_colors_str[0]]*len(trials_start_times)+
                    [trials_boundaries_colors_str[1]]*len(trials_end_times))
    trial_boundaries_df = pd.DataFrame(dict(event_time=bound_times,
                                            event_line_type=bound_linetypes,
                                            event_color=bound_colors)
                                      )

    plotUtils.add_events_vlines(fig=fig, events_df=trial_boundaries_df)
    title = f"Lower bound: {final_lower_bound:.02f}"
    fig.update_layout(title=title)
    fig.write_image(latents_fig_filename_pattern.format("png"))
    fig.write_html(latents_fig_filename_pattern.format("html"))

    print(f'Image saved to {latents_fig_filename_pattern.format("html")}')

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
