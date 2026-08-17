import sys
import argparse
import plotly.graph_objects as go

import svGPFA.utils.miscUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_latents", type=int, default=3,
                        help="number of latent variables")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="number of trials")
    parser.add_argument("--n_ind_points", type=int, default=5,
                        help="number of inducing points")
    parser.add_argument("--n_neurons", type=int, default=50,
                        help="number of neurons")
    parser.add_argument("--n_params_all_kernels", type=int, default=3,
                        help="number of parameters of all kernels")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/num_params_nLatents{:d}_nTrials{:d}_nIndPoints{:d}_nNeurons{:d}_nParamsAllKernels{:d}.{{:s}}",
                        help="figure filename pattern")
    args = parser.parse_args()

    n_latents = args.n_latents
    n_trials = args.n_trials
    n_ind_points = args.n_ind_points
    n_neurons = args.n_neurons
    n_params_all_kernels = args.n_params_all_kernels
    fig_filename_pattern = args.fig_filename_pattern.format(
        n_latents, n_trials, n_ind_points, n_neurons, n_params_all_kernels)

    n_params = svGPFA.utils.miscUtils.countParams(n_latents=n_latents,
                                                   n_trials=n_trials,
                                                   n_ind_points=n_ind_points,
                                                   n_neurons=n_neurons,
                                                   n_params_all_kernels=n_params_all_kernels)

    x = tuple(n_params.keys())
    y = tuple(n_params.values())
    x = x + ("total",)
    y = y + (sum(y),)

    fig = go.Figure()

    trace = go.Bar(x=x, y=y)
    fig.add_trace(trace)
    fig.update_yaxes(title="Number of Parameters")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved image {:s}".format(fig_filename_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
