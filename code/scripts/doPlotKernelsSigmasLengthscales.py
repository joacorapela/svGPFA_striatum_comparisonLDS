
import sys
import pickle
import argparse
import jax.numpy as jnp
import plotly.graph_objects as go

import svGPFA.plot.plotUtilsPlotly

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        # default=23521323)
                        default=30222892)
                        # default=54368807)
    parser.add_argument("--est_results_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_kernels_sigmas_lengthscales.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    est_res_filename = args.est_results_filename_pattern.format(
        est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(est_res_filename, "rb") as f:
        est_res = pickle.load(f)
    kernels_params = est_res["estimated_params"]["kernels_params"]

    kernels_sigmas = [params_pair[0] for params_pair in kernels_params]
    kernels_lengthscales = [params_pair[1] for params_pair in kernels_params]

    kernels_indices = jnp.arange(len(kernels_sigmas))

    fig = go.Figure()
    trace = go.Bar(x=kernels_indices, y=kernels_sigmas, name="Sigmas")
    fig.add_trace(trace)
    trace = go.Bar(x=kernels_indices, y=kernels_lengthscales, name="Lengthscales")
    fig.add_trace(trace)
    fig.update_xaxes(title="Kernel/Latent Index")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved image {:s}".format(fig_filename_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
