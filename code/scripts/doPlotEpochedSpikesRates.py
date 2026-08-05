
import sys
import argparse
import pickle

import gcnu_common.utils
import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_number", type=int, help="epoch number",
                        default=96439322)
    parser.add_argument("--subject_implant_label", type=str,
                        help="subject and implant name",
                        default="EJT178_implant1")
    parser.add_argument("--recording_label", type=str, help="recording label",
                        default="recording6_29-03-2022")
                        # default="recording7_30-03-2022")
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default="../../../svGPFA_striatum/results/{:s}/{:s}/{:08d}_epoched_spikes_times.pickle")
                        # default="../../../svGPFA_striatum/results/EJT178_implant1/recording6_29-03-2022/47205531_pseudo_epoched_spikes_times.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help=("figure filename pattern"),
                        type=str,
                        default=("../../figures/{:s}/{:s}/spikes_rates_{:08d}.{{:s}}"))
    args = parser.parse_args()

    subject_implant_label = args.subject_implant_label
    recording_label = args.recording_label
    epoch_number = args.epoch_number
    epoched_spikes_times_filename = \
        args.epoched_spikes_times_filename_pattern.format(
            subject_implant_label, recording_label, epoch_number)
    fig_filename_pattern = args.fig_filename_pattern.format(
        subject_implant_label, recording_label, epoch_number)

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    clusters_ids = load_res["clusters_ids"]
    trials_ids = load_res["trials_ids"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]
    n_trials = len(spikes_times)

    trials_durations = [trials_end_times[r] - trials_start_times[r]
                        for r in range(n_trials)]
    spikes_rates_allTrials_allNeurons = \
        gcnu_common.utils.neural_data_analysis.\
        getSpikesRatesAllTrialsAllClusters(
            spikes_times=spikes_times, trials_durations=trials_durations)

    fig = svGPFA.plot.plotUtilsPlotly.\
        getPlotSpikesRatesAllTrialsAllClusters(
            spikes_rates=spikes_rates_allTrials_allNeurons,
            trials_ids=trials_ids, clusters_ids=clusters_ids)

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print(f'Saved {fig_filename_pattern.format("html")}')

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
