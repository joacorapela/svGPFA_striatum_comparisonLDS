
import sys
import warnings
import argparse
import configparser
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle
import itertools
import plotly.graph_objects as go

import svGPFA.utils.miscUtils
import svGPFA.utils.statsUtils
import striatumUtils
import plotUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=91676545)
                        # default=35179010)
                        # default=69706576)
                        # default=38426992)
                        # default=97656976)
                        # default=20731399)
                        # default=96281561)
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
    parser.add_argument("--estimation_res_filename_pattern",
                        help="estimation results filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    # parser.add_argument("--inferred_model_filename_pattern",
    #                     help="saved inferred model filename pattern", type=str,
    #                     default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
    parser.add_argument("--latents_fig_filename_pattern",
                        help="latents figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_orthonormalized_nonEpoched_latents.{{:s}}")
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
    estimation_res_filename_pattern = args.estimation_res_filename_pattern
    # inferred_model_filename_pattern = args.inferred_model_filename_pattern
    latents_fig_filename_pattern = args.latents_fig_filename_pattern.format(est_res_number)

    ports_markers = dict(zip(ports_to_plot, ports_markers_str))
    in_ports_linetypes = dict(zip(ports_to_plot, in_ports_linetypes_str))
    out_ports_linetypes = dict(zip(ports_to_plot, out_ports_linetypes_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    estimation_res_filename = estimation_res_filename_pattern.format(est_res_number)

    with open(estimation_res_filename, "rb") as f:
        estimation_res = pickle.load(f)
    final_lower_bound = estimation_res["lower_bound_hist"][-1]
    trials_ids = estimation_res["trials_ids"].tolist()
    kernels_types = estimation_res["kernels_types"]
    clusters = estimation_res["selected_clusters"]
    leg_quad_points = estimation_res["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    reg_param = 1e-5
    estimated_params = estimation_res["estimated_params"]
    if inferred:
        fixed_params = estimation_res["fixed_params"]
        # fixed_params = estimation_res["fixed_params"][0]
    trials_start_times = estimation_res["trials_start_times"]
    trials_end_times = estimation_res["trials_end_times"]
    epochs_times = estimation_res["epochs_times"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    # kernels_params = estimated_params["kernels_params"]
    if inferred:
        C = fixed_params["C"]
        d = fixed_params["d"]
        kernels_params = fixed_params["kernels_params"]
    else:
        C = estimated_params["C"]
        d = estimated_params["d"]
        kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

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
    times_non_epoched = striatumUtils.get_times_non_epoched(
        times=np.array(times), epochs_times=np.array(epochs_times))
    ol_means_non_epoched = striatumUtils.get_latents_non_epoched(latents=ol_means)
    ol_vars_non_epoched = striatumUtils.get_latents_non_epoched(latents=ol_vars)

    n_time_points_per_trial = l_means.shape[1]
    trials_ids_for_samples = list(itertools.chain.from_iterable(
        ([trial_id] * n_time_points_per_trial for trial_id in trials_ids)
    ))

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
        latents_colors_patterns=latents_colors_patterns,
        trials_ids_for_samples=trials_ids_for_samples,
        xlabel="Time (sec)")

    ports_events_df = plotUtils.build_events_df(
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        transition_data=transition_data,
        in_ports_linetypes=in_ports_linetypes,
        out_ports_linetypes=out_ports_linetypes,
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
