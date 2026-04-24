
import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
    parser.add_argument("--hmm_params_filename_pattern", type=str,
                        help="hmm parameters filename pattern",
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_params.pickle")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_hmm_{{:s}}.{{:s}}")

    args = parser.parse_args()

    est_res_number = args.est_res_number
    hmm_params_filename = args.hmm_params_filename_pattern.format(est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(hmm_params_filename, "rb") as f:
        load_res = pickle.load(f)
    A = load_res["A"] # \in (K, K)
    means = load_res["means"] # \in (K, 1, R)
    covs = load_res["covs"] # \in (K, K, R)
    R = means.shape[2]

    latents = np.arange(R)
    stds = np.sqrt(np.diagonal(covs, axis1=0, axis2=1)).T

    fig = go.Figure()
    trace = go.Heatmap(z=A)
    fig.add_trace(trace)
    fig.update_xaxes(title="To State")
    fig.update_yaxes(title="From State")
    fig.write_image(fig_filename_pattern.format("transition", "png"))
    fig.write_html(fig_filename_pattern.format("transition", "html"))
    print(f'saved {fig_filename_pattern.format("transition", "html")}')
    # fig.show()

    fig = go.Figure()
    for i in range(R):
        trace = go.Bar(x=latents, y=means[:, 0, i], name=f"State {i}",
                       error_y=dict(type='data', array=stds[:, i]))
        fig.add_trace(trace)
    fig.update_xaxes(title="Latent Index")
    fig.update_yaxes(title=r"$\text{Mean}\pm\text{STD}$")
    fig.write_image(fig_filename_pattern.format("mean", "png"))
    fig.write_html(fig_filename_pattern.format("mean", "html"))
    print(f'saved {fig_filename_pattern.format("mean", "html")}')

    # fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
