
import sys
import pickle
import argparse
import jax.flatten_util
import jax.numpy as jnp
import plotly.graph_objects as go


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
                        # default=33576128)
    parser.add_argument("--est_res_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_kernels_params_derivatives.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    est_res_filename = args.est_res_filename_pattern.format(args.est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(est_res_filename, "rb") as f:
        est_res = pickle.load(f)

    if "state" in est_res:
        grad = est_res["state"].grad
    elif "estimated_state" in est_res:
        grad = est_res["estimated_state"].grad
    else:
        raise ValueError("state or estimated_state should be keys of est_res")

    kernels_params_derivatives = grad["kernels_params"]
    n_latents = len(kernels_params_derivatives)

    fig = go.Figure()
    for k in range(n_latents):
        trace = go.Bar(x=[kernels_params_derivatives[k].item()], y=[1], name=f"latent {k}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Derivative")
    fig.update_yaxes(title="Count")
    fig.update_layout(
        title=(f"Number of Parameters: {len(kernels_params_derivatives)}"))

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved image {:s}".format(fig_filename_pattern.format("html")))

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
