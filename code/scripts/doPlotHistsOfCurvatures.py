
import sys
import pickle
import argparse
import jax
import jax.numpy as jnp
import plotly.graph_objects as go


def get_plot_histogram_curvatures(curvatures, title=None,
                                  percentiles=None,
                                  xlabel="Curvature",
                                  ylabel="Count"):
    fig = go.Figure()
    trace = go.Histogram(x=curvatures)
    fig.add_trace(trace)
    if percentiles is not None:
        fig.update_xaxes(title=xlabel, range=percentiles)
    else:
        fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    if title is not None:
        fig.update_layout(title=title)
    return fig


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=79151056)
    parser.add_argument("--hessian_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hessian.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures_{{:s}}.{{{{:s}}}}",
                       )
    args = parser.parse_args()

    est_res_number = args.est_res_number
    hessian_filename = args.hessian_filename_pattern.format(
        est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(hessian_filename, "rb") as f:
        hessian = pickle.load(f)

    n_kernels_params = len(hessian["kernels_params"])
    hessian_keys = [*hessian]

    # kernels_params are lists and should be treated differently
    hessian_keys.remove("kernels_params")

    curvatures = dict()

    kernels_params_hessian = hessian["kernels_params"]
    curvatures["kernels_params"] = jnp.array([kernels_params_hessian[i]["kernels_params"][i].item() for i in range(len(kernels_params_hessian))])

    # process other parameters
    for i in range(len(hessian_keys)):
        key = hessian_keys[i]
        hessian_i = hessian[key][key]

        # Deduce original parameter shape from the first half of hessian_i's dimensions
        ndim = hessian_i.ndim // 2
        param_shape = hessian_i.shape[:ndim]

        # Compute total number of scalar elements in this parameter
        num_elements = jnp.prod(jnp.array(param_shape))

        # Flatten the (N + N)-dimensional block into a (K, K) 2D matrix
        flat_hessian = hessian_i.reshape(num_elements, num_elements)

        # Take the 1D diagonal
        flat_diag = jnp.diag(flat_hessian)
        # Reshape back to the parameter's original N-dimensional shape
        curvatures[key] = flat_diag.reshape(param_shape)

    # build plots
    for key in curvatures:
        if key=="kernels_params":
            fig = go.Figure()
            trace = go.Scatter(x=curvatures[key],
                               y=jnp.ones(len(curvatures[key])),
                               mode="markers")
            fig.add_trace(trace)
            fig.update_xaxes(title="Curvature")
            fig.update_yaxes(title="Count")
        else:
            percentiles = jnp.percentile(curvatures[key], jnp.array([1, 99]))
            fig = get_plot_histogram_curvatures(
                    curvatures=curvatures[key].flatten(),
                    percentiles=percentiles)

        fig_filename_with_keys_pattern = fig_filename_pattern.format(key)
        fig.write_image(fig_filename_with_keys_pattern.format("png"))
        fig.write_html(fig_filename_with_keys_pattern.format("html"))
        print("Saved {:s}".format(
            fig_filename_with_keys_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
