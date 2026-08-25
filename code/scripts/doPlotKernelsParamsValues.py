
import sys
import pickle
import argparse
import jax.numpy as jnp
import plotly.graph_objects as go


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        # default=23521323)
                        default=54368807)
    parser.add_argument("--est_results_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_kernels_params_values.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    est_res_filename = args.est_results_filename_pattern.format(
        est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(est_res_filename, "rb") as f:
        est_res = pickle.load(f)
    kernels_params_values = [value.item() for value in est_res["estimated_params"]["kernels_params"]]
    n_latents = len(kernels_params_values)

    fig = go.Figure()
    trace = go.Bar(x=jnp.arange(n_latents),
                   y=kernels_params_values)
    fig.add_trace(trace)
    fig.update_xaxes(title="Latent Index")
    fig.update_yaxes(title="Kernels Lengthscales")
    fig.update_layout(title=f"Number of Parameters: {len(kernels_params_values)}")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved image {:s}".format(fig_filename_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
