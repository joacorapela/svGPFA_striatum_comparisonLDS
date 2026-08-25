
import sys
import json
import jax.numpy as jnp
import jax.tree
import pickle
import argparse
import plotly.graph_objects as go


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_numbers", help="models est_res_number",
                        type=str, default="[54368807,32039350,52098333,63850714,17983794,57166803,87935199,37180679,44604524,67518766,83227089]")
    parser.add_argument("--curvatures_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:s}_max_abs_curvature.{{:s}}")
    args = parser.parse_args()

    est_res_numbers = json.loads(args.est_res_numbers)
    curvatures_filename_pattern = args.curvatures_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern.format(
        args.est_res_numbers[1:-1].replace(",","_"))

    max_abs_curvatures = []
    for est_res_number in est_res_numbers:
        curvatures_filename = curvatures_filename_pattern.format(est_res_number)
        with open(curvatures_filename, "rb") as f:
            curvatures = pickle.load(f)
        max_abs_curvature = jax.tree.reduce(jnp.maximum, jax.tree.map(lambda x: jnp.max(jnp.abs(x)), curvatures))
        max_abs_curvatures.append(max_abs_curvature)
        print(f"{est_res_number}: max_abs_curvature={max_abs_curvature}")

    est_res_numbers_str = [str(est_res_number) for est_res_number in
                           est_res_numbers]
    fig = go.Figure()
    trace = go.Scatter(x=est_res_numbers_str, y=max_abs_curvatures)
    fig.add_trace(trace)

    fig.update_xaxes(
        title_text='Model ID',
        type='category',
        categoryorder='array',
        categoryarray=est_res_numbers_str,
        tickangle=-90  # Rotates labels 90 degrees vertically
    )

    fig.update_yaxes(title_text="Maximum Curvature")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved {:s}".format(fig_filename_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
