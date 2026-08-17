
import sys
import pickle
import argparse
import jax.numpy as jnp
import plotly.graph_objects as go


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
                        # default=33576128)
    parser.add_argument("--curvatures_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_ind_points_locs_curvatures.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    curvatures_filename = args.curvatures_filename_pattern.format(
		est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(curvatures_filename, "rb") as f:
        curvatures = pickle.load(f)

    ind_points_locs_curvatures = curvatures["ind_points_locs"]
    n_latents = ind_points_locs_curvatures.shape[0]
    [perc_1, perc_99] = jnp.percentile(ind_points_locs_curvatures, jnp.array([1, 99]))

    fig = go.Figure()
    for k in range(n_latents):
        trace = go.Histogram(x=ind_points_locs_curvatures[k, :, :, 0].flatten(), name=f"latent {k}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Curvature", range=(perc_1, perc_99))
    fig.update_yaxes(title="Count")
    fig.update_layout(title=f"Number of Parameters: {ind_points_locs_curvatures.size}")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved image {:s}".format(fig_filename_pattern.format("html")))

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
