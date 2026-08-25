
import sys
import pickle
import argparse
import jax
import numpy as np
import plotly.graph_objects as go


def get_plot_histogram_hessian_coefs(coefs, title,
                                     x_percentiles=None,
                                     xlabel="Hessian Coefficient",
                                     ylabel="Count"):
    coefs = np.array(coefs) # convert from jax to numpy

    coefs_nan_removed = coefs.copy()
    if x_percentiles is not None:
        coefs = np.clip(coefs, a_min=x_percentiles[0], a_max=x_percentiles[1])

    # print(title)
    # breakpoint()

    fig = go.Figure()
    trace = go.Histogram(x=coefs)
    fig.add_trace(trace)
    if x_percentiles is not None:
        fig.update_xaxes(title=xlabel, range=x_percentiles)
    else:
        fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
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
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_hessian_coefs_{{:s}}_{{:s}}.{{{{:s}}}}",
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

    # process kernels_params with kernels params
    key1 = "kernels_params"
    key2 = "kernels_params"
    flat_hessian_kernels_coefs_list = []
    for i in range(n_kernels_params):
        for j in range(i, n_kernels_params):
            flat_hessian_kernels_coefs_list.append(hessian[key1][i][key2][j].item())
    fig_filename_with_keys_pattern = fig_filename_pattern.format(key1, key2)
    title = f"{key1} -- {key2}"

    fig = go.Figure()
    trace = go.Scatter(x=flat_hessian_kernels_coefs_list,
                       y=np.ones(len(flat_hessian_kernels_coefs_list)),
                       mode="markers")
    fig.add_trace(trace)
    fig.update_layout(title=title)
    fig.update_xaxes(title="Hessian Coefficient")
    fig.update_yaxes(title="Count")
    fig.update_layout(
        title=f"{key1} -- {key2}, Number of Parameters: {len(flat_hessian_kernels_coefs_list)}")

    fig.write_image(fig_filename_with_keys_pattern.format("png"))
    fig.write_html(fig_filename_with_keys_pattern.format("html"))
    print("Saved {:s}".format(
        fig_filename_with_keys_pattern.format("html")))

    # process kernels_params with other params
    key1 = "kernels_params"
    for key2 in hessian_keys:
        flat_hessian_coefs_list = []
        for i in range(n_kernels_params):
            flat_hessian_coefs_list += [x.ravel() for x in hessian[key1][i][key2]]
        flat_hessian_coefs = np.concatenate(flat_hessian_coefs_list)
        fig_filename_with_keys_pattern = fig_filename_pattern.format(key1, key2)
        x_percentiles = np.percentile(flat_hessian_coefs, np.array([1, 99]))
        fig = get_plot_histogram_hessian_coefs(
            coefs=flat_hessian_coefs,
            title=f"{key1} -- {key2}, Number of Parameters: {len(flat_hessian_coefs)}",
            x_percentiles=x_percentiles)
        fig.write_image(fig_filename_with_keys_pattern.format("png"))
        fig.write_html(fig_filename_with_keys_pattern.format("html"))
        print("Saved {:s}".format(
                fig_filename_with_keys_pattern.format("html")))

    # process other parameters
    for i in range(len(hessian_keys)):
        key1 = hessian_keys[i]
        for j in range(i, len(hessian_keys)):
            key2 = hessian_keys[j]
            flat_hessian_coefs = hessian[key1][key2].flatten()
            fig_filename_with_keys_pattern = fig_filename_pattern.format(key1,
                                                                         key2)
            x_percentiles = np.percentile(flat_hessian_coefs, np.array([1, 99]))
            fig = get_plot_histogram_hessian_coefs(
                coefs=flat_hessian_coefs,
                title=f"{key1} -- {key2}, Number of Parameters: {len(flat_hessian_coefs)}",
                x_percentiles=x_percentiles)
            fig.write_image(fig_filename_with_keys_pattern.format("png"))
            fig.write_html(fig_filename_with_keys_pattern.format("html"))
            print("Saved {:s}".format(
                fig_filename_with_keys_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
