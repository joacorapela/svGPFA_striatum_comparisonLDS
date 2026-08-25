
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
                        type=str, default="[54368807,32039350,52098333,63850714,17983794,57166803,87935199,83227089]")
    parser.add_argument("--curvatures_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_curvatures.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:s}_max_curvature.{{:s}}")
    args = parser.parse_args()

    est_res_numbers = json.loads(args.est_res_numbers)
    curvatures_filename_pattern = args.curvatures_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern.format(
        args.est_res_numbers[1:-1].replace(",","_"))

    diag_cond_numbers = []
    for est_res_number in est_res_numbers:
        curvatures_filename = curvatures_filename_pattern.format(est_res_number)
        with open(curvatures_filename, "rb") as f:
            curvatures = pickle.load(f)
        min_cond_number = jax.tree.reduce(jnp.minimum, jax.tree.map(lambda x: jnp.min(jnp.abs(x)), curvatures))
        max_cond_number = jax.tree.reduce(jnp.maximum, jax.tree.map(lambda x: jnp.max(jnp.abs(x)), curvatures))
        diag_cond_number = max_cond_number / min_cond_number
        diag_cond_numbers.append(diag_cond_number)
        print(f"{est_res_number}: min={min_cond_number}, max={max_cond_number}, diag_cond_number={diag_cond_number}")

    est_res_numbers_str = [str(est_res_number) for est_res_number in
                           est_res_numbers]
    fig = go.Figure()
    trace = go.Scatter(x=est_res_numbers_str, y=diag_cond_numbers)
    fig.add_trace(trace)

    fig.update_xaxes(
        title_text='Model ID',
        type='category',
        categoryorder='array',
        categoryarray=est_res_numbers_str,
        tickangle=-90  # Rotates labels 90 degrees vertically
    )

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved {:s}".format(fig_filename_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
