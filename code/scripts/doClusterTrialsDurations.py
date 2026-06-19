
import sys
import os.path
import argparse
import random
import pickle
import numpy as np
import scipy.stats
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots

# import vbgmm
from sklearn.mixture import BayesianGaussianMixture

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_implant_label", type=str,
                        help="subject and implant name",
                        default="EJT178_implant1")
    parser.add_argument("--recording_label", type=str, help="recording label",
                        default="recording6_29-03-2022")
                        # default="recording7_30-03-2022")
    parser.add_argument("--k", type=int, help="number of clusters",
                        default=2)
    parser.add_argument("--collapsed_cluster_thr", type=float,
                        help="an active cluster should have a weight greater than collapsed_cluster_thr",
                        default=1e-4)
    parser.add_argument("--skip_log_transform", action="store_true",
                        help="select this option to transform trials durations to log scale")
    parser.add_argument("--trials_times_filename_pattern", type=str,
                        help="trials times filename",
                        default="/nfs/gatsbystor/rapela/work/ucl/gatsby-swc/collaborations/emmett/repo/results/{:s}/{:s}/trials_times.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/{:s}/{:s}/{{:08d}}_trials_times_clustering_result.{{:s}}")
    parser.add_argument("--clustering_res_filename_pattern", type=str,
                        help="clustering result filename pattern",
                        default="../../results/{:s}/{:s}/{{:08d}}_trials_times_clustering_result.{{:s}}")
    args = parser.parse_args()

    subject_implant_label = args.subject_implant_label
    recording_label = args.recording_label
    k = args.k
    collapsed_cluster_thr = args.collapsed_cluster_thr
    skip_log_transform = args.skip_log_transform
    trials_times_filename = args.trials_times_filename_pattern.format(
        subject_implant_label, recording_label)
    fig_filename_pattern = args.fig_filename_pattern.format(
        subject_implant_label, recording_label)
    clustering_res_filename_pattern = \
        args.clustering_res_filename_pattern.format(subject_implant_label,
                                                    recording_label)

    trials_times = pd.read_csv(trials_times_filename)
    trials_durations = trials_times["end_time"] - trials_times["start_time"]

    if not skip_log_transform:
        data = np.log(trials_durations)
    else:
        data = trials_durations

    data = np.expand_dims(data, 1)

    # Normalization for stability
    data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
    data_norm = (data - data_mean) / data_std

    # 2. Run the Estimation (Using the modular functions from before)
    # results = vbgmm.estimate(data_norm, k=k, verbose=True)

    # res = results["responsibilities"]
    # means = results["means"]
    # covariances = results["covariances"]
    # weights = results["weights"]
    # elbos = results["elbos"]

    # bgmm = BayesianGaussianMixture(n_components=2, random_state=42)
    bgmm = BayesianGaussianMixture(n_components=2)
    bgmm.fit(data_norm)

    # build modelSaveFilename
    prefix_used = True
    while prefix_used:
        clustering_res_number = random.randint(0, 10**8)
        clustering_res_metadata_filename = \
            clustering_res_filename_pattern.format(clustering_res_number, "ini")
        if not os.path.exists(clustering_res_metadata_filename):
            prefix_used = False
    clustering_res_filename = clustering_res_filename_pattern.format(
        clustering_res_number, "pickle")

    # This returns an array of shape (N_trials, 2)
    rs = bgmm.predict_proba(data_norm)
    means = bgmm.means_
    covariances = bgmm.covariances_
    weights = bgmm.weights_
    elbos = bgmm.lower_bounds_

    results = dict(
        rs = rs,
        means = means,
        covariances = covariances,
        weights = weights,
        elbos = elbos,
    )

    with open(clustering_res_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Clustering result saved to {clustering_res_filename}.")

    active_indices = np.where(weights > collapsed_cluster_thr)[0]
    D = data.shape[1]

    # Plot results
    fig = plotly.subplots.make_subplots(
        rows=1, cols=2,
        subplot_titles=("Lower Bound Convergence (ELBO)", f"Active Clusters: {len(active_indices)}")
    )

    # --- Plot 1: ELBO (CB 10.70) ---
    fig.add_trace(
        go.Scatter(x=list(range(len(elbos))), y=elbos, mode='lines+markers', name='ELBO'),
        row=1, col=1
    )

    # --- Plot 2: Data & Clusters ---
    # Background Data Points
    if skip_log_transform:
        x_label = "Trials Durations (secs)"
    else:
        x_label = "Log Trials Durations (log secs)"
    fig.add_trace(
        go.Histogram(x=data[:, 0],
                     histnorm="probability density",
                     name="Data Histogram",
                    ),
                     # histnorm="density"),
                     # histnorm="probability"),
        row=1, col=2
    )

    min_trial_duration = np.min(data)
    max_trial_duration = np.max(data)
    n_x_dense = 1000
    x_dense = np.linspace(min_trial_duration, max_trial_duration, n_x_dense)

    for idx in active_indices:
        # Expected Covariance calculation (CB B.82)
        # solve((nuk-D-1)*wk) equivalent
        nuk = D + np.sum(rs[:, idx])
        # wk = covariances[:, :, idx]
        wk = covariances[idx, :, :]

        if nuk > D + 1:
            cov_norm = np.linalg.inv((nuk - D - 1) * wk)
        else:
            cov_norm = np.linalg.inv(wk)

        # Re-scale back to original units
        # mu = (means[:, idx] * data_std) + data_mean
        mu = (means[idx, :] * data_std) + data_mean
        cov = cov_norm * (data_std**2)

        y = scipy.stats.norm.pdf(x_dense, loc=mu[0], scale=np.sqrt(cov[0,0]))

        # Add Cluster pdf
        label = fr"$\text{{Cluster }} {idx}\quad(\pi_{idx} = {weights[idx]:.2f})$"
        fig.add_trace(
            go.Scatter(x=x_dense, y=y, mode='lines', line=dict(width=2),
                       name=label),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="ELBO", row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)

    fig.write_html(fig_filename_pattern.format(clustering_res_number, "html"))
    fig.write_image(fig_filename_pattern.format(clustering_res_number, "png"))
    print(f'Figure saved to {fig_filename_pattern.format(clustering_res_number, "html")}')

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
