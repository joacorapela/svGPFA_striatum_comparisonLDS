
import sys
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import hmmUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_number",
                        help="Viterbi run result number",
                        type=int,
                        default=88250367)
                        # default=63183665)
                        # default=83727406)
    parser.add_argument("--port_label_col_name",
                        help="column name for port label",
                        type=str, default="Start_Port")
    parser.add_argument("--port_enter_times_col_name",
                        help="column name for port enter (ephys) time",
                        type=str, default="P1_IN_Ephys_TS")
    parser.add_argument("--port_exit_times_col_name",
                        help="column name for port exit (ephys) time",
                        type=str, default="P1_OUT_Ephys_TS")
    parser.add_argument("--est_results_filename_pattern",
                        help="estimation results filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--transitions_data_filename",
                        help="transition data filename", type=str,
                        default="/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording6_29-03-2022/behav_sync/2_task/Transition_data_sync.csv")
    parser.add_argument("--most_prob_states_seq_filename_pattern", type=str,
                        help="most probable states sequence filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_most_prob_state_seq.{:s}")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_most_prob_state_seq.{:s}")

    args = parser.parse_args()

    res_number = args.res_number
    port_label_col_name = args.port_label_col_name
    port_enter_times_col_name = args.port_enter_times_col_name
    port_exit_times_col_name = args.port_exit_times_col_name
    est_results_filename_pattern = args.est_results_filename_pattern
    transitions_data_filename = args.transitions_data_filename
    most_prob_states_seq_filename_pattern = args.most_prob_states_seq_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    metadata_filename = most_prob_states_seq_filename_pattern.format(res_number,
                                                                     "metadata")
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    test_est_res_number = int(metadata["params"]["test_est_res_number"])

    est_results_filename = est_results_filename_pattern.format(test_est_res_number)
    with open(est_results_filename, "rb") as f:
        est_results = pickle.load(f)
    epochs_times = est_results["epochs_times"]

    most_prob_states_seq_filename = \
        most_prob_states_seq_filename_pattern.format(res_number, "pickle")
    with open(most_prob_states_seq_filename, "rb") as f:
        viterbi_res = pickle.load(f)
    trials_times = viterbi_res["trials_times"]
    most_prob_states_seq = viterbi_res["most_prob_states_seq"]

    # build continuous_time
    T = len(trials_times)
    continuous_times_list = []
    for t in range(T):
        continuous_times_list.extend(trials_times[t][:, 0] + epochs_times[t])
    continuous_times = np.array(continuous_times_list)

    # build continuous_most_prob_states_seq
    continuous_most_prob_states_seq = np.concatenate(most_prob_states_seq)

    # test_start_time_sec = continuous_times.min()
    # test_end_time_sec = continuous_times.max()

    # transitions_data = pd.read_csv(transitions_data_filename)

    # in_range_bool = np.logical_and(
    #     test_start_time_sec<=transitions_data[port_enter_times_col_name],
    #     transitions_data[port_exit_times_col_name]<test_end_time_sec)
    # transitions_data = transitions_data[in_range_bool]
    # port_labels = transitions_data[port_label_col_name].to_numpy()
    # enter_times = transitions_data[port_enter_times_col_name].to_numpy()
    # exit_times = transitions_data[port_exit_times_col_name].to_numpy()

    # states_labels_true, states_seq_true = hmmUtils.getFilteredBehavioralStates(
    #     bins_centers=continuous_times,
    #     port_labels=port_labels,
    #     enter_times=enter_times,
    #     exit_times=exit_times)

    fig = go.Figure()
    trace = go.Scatter(x=continuous_times, y=continuous_most_prob_states_seq, name="Inferred", mode="lines+markers")
    fig.add_trace(trace)
    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="State Number")
    html_filename = fig_filename_pattern.format(res_number, "html")
    fig.write_html(html_filename)
    png_filename = fig_filename_pattern.format(res_number, "png")
    fig.write_image(png_filename)

    print(f"figure saved to {html_filename}")

    # fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
