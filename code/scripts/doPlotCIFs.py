
import sys
import configparser
import numpy as np
import pandas as pd
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
import jax.numpy as jnp
import pickle
import argparse
import plotly.colors

import svGPFA.utils.statsUtils
import svGPFA.plot.plotUtilsPlotly

import striatumUtils
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
    parser.add_argument("--cluster_id", help="cluster ID to plot", type=int,
                        default=132)
    parser.add_argument("--bin_size_secs", help="bin size (secs)",
                        type=float, default=0.01)
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
    parser.add_argument("--model_filename_pattern",
                        help="model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--transitions_data_filename",
                        help="transitions data filename",
                        type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_CIF_clusterID{:03d}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    cluster_id = args.cluster_id
    bin_size_secs = args.bin_size_secs
    ports_to_plot = [int(port_str) for port_str in args.ports_to_plot.split(",")]
    ports_markers_str = args.ports_markers_str.split(",")
    ports_colors_str = args.ports_colors_str.split(",")
    model_filename = args.model_filename_pattern.format(est_res_number)
    metadata_filename = args.metadata_filename_pattern.format(est_res_number)
    transitions_data_filename = args.transitions_data_filename
    fig_filename_pattern = args.fig_filename_pattern.format(
        est_res_number, cluster_id)

    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    epoched_spikes_times_filename = metadata["data_params"]["epoched_spikes_times_filename"]

    # get spike times
    with open(epoched_spikes_times_filename, "rb") as f:
        epochs_res = pickle.load(f)
    spikes_times = epochs_res["spikes_times"]
    trials_ids = epochs_res["trials_ids"]
    trials_start_times = np.array(epochs_res["trials_start_times"])
    trials_end_times = np.array(epochs_res["trials_end_times"])
    epochs_times = np.array(epochs_res["epochs_times"])

    # get parameters from estimated model
    with open(model_filename, "rb") as f:
        est_results = pickle.load(f)
    if "lower_bound_hist" in est_results:
        final_lower_bound = est_results["lower_bound_hist"][-1]
    else:
        final_lower_bound = -est_results["estimated_state"].value.item()
    if "selected_trials_ids" in est_results:
        selected_trials_ids = est_results["selected_trials_ids"].tolist()
    elif "trials_ids" in est_results:
        selected_trials_ids = est_results["trials_ids"].tolist()
    else:
        RuntimeError("either selected_trials_ids or trials_ids should exist in est_results")
    kernels_types = est_results["kernels_types"]
    clusters_ids = est_results["clusters_ids"]
    selected_clusters = est_results["selected_clusters"]

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

    transitions_data = pd.read_csv(transitions_data_filename)

    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    # reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    reg_param = 1e-5
    estimated_params = est_results["estimated_params"]
    fixed_params = est_results["fixed_params"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    if "C" in estimated_params and "d" in estimated_params:
        C = estimated_params["C"]
        d = estimated_params["d"]
    elif "C" in fixed_params and "d" in fixed_params:
        C = fixed_params["C"]
        d = fixed_params["d"]
    else:
        raise RuntimeError("C and d could not be found in estimated_params or fixed_params")
    kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    times = jnp.asarray(leg_quad_points)

    assert(len(selected_clusters) == C.shape[0])
    cluster_index = np.where(selected_clusters == cluster_id)[0][0]

    # build markers and trials colors
    ports_markers = dict(zip(ports_to_plot, ports_markers_str))
    ports_colors = dict(zip(ports_to_plot, ports_colors_str))

    marked_events_times, marked_events_colors, marked_events_markers, \
            marked_events_labels = plotUtils.buildMarkedEventsInfoFromTransitions(
                transitions_data=transitions_data, trials_ids=selected_trials_ids)

    events_names = ["trial {:d}".format(trial_id)
                    for trial_id in selected_trials_ids]
                    # for trial_id in marked_events_trials_ids]

    n_trials = len(trials_ids)
    colorscale = plotly.colors.qualitative.Light24
    colorscale_rgb = [plotly.colors.hex_to_rgb(hex_color)
                      for hex_color in colorscale]
    colors_patterns = ["rgba({:d},{:d},{:d},{{:f}})".format(color[0], color[1], color[2]) for color in colorscale_rgb]
    trials_colors_patterns = [colors_patterns[r%len(colors_patterns)] for r in range(n_trials)]
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    # compute CIF
    h_means, h_vars = svGPFA.utils.statsUtils.computePreIntensity(
        C=C, d=d, vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    cif_means, cif_stds = svGPFA.utils.statsUtils.computeCIFmeanAndSTD(
        h_means=h_means, h_vars=h_vars)

    title = f"Lower bound: {final_lower_bound:.02f}"

    # plot CIF
    title = f"Cluster ID: {cluster_id}"
    fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneClusterAllTrials(
        trials_times=times,
        cif_means=cif_means,
        cif_stds=cif_stds,
        cluster_index=cluster_index,
        bin_size_secs=bin_size_secs,
        spikes_times=spikes_times,
        trials_ids=selected_trials_ids,
        align_event_times=epochs_times,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors_patterns=trials_colors_patterns,
        trials_colors=trials_colors,
        title=title,
    )
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved {:s}".format(fig_filename_pattern.format("html")))

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
