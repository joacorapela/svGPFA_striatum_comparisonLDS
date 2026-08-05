
import sys
import configparser
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
import pickle
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=93116083)
    parser.add_argument("--field", help="estimation results field to plot",
                        type=str, default="error")
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_filename_pattern",
                        help="model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_state_field_{:s}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    field = args.field
    metadata_filename = args.metadata_filename_pattern.format(est_res_number)
    model_filename_pattern = args.model_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number, field)

    print(f"Processing model {est_res_number}")
    est_res_numbers = [est_res_number]
    model_filename = model_filename_pattern.format(est_res_number)
    with open(model_filename, "rb") as f:
        est_res = pickle.load(f)
    field_attrs = [getattr(est_res["estimated_state"], field)]

    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    estimation_params = dict(metadata["estimation_params"])
    exit_loop = False
    if not "in_est_res_number" in estimation_params:
        exit_loop = True

    while not exit_loop:
        est_res_number = int(estimation_params["in_est_res_number"])
        print(f"Processing model {est_res_number}")
        est_res_numbers.insert(0, est_res_number)
        model_filename = model_filename_pattern.format(est_res_number)
        with open(model_filename, "rb") as f:
            est_res = pickle.load(f)
            if "estimated_state" in est_res:
                field_attrs.insert(0, getattr(est_res["estimated_state"], field))
            elif "state" in est_res:
                field_attrs.insert(0, getattr(est_res["state"], field))
            else:
                RuntimeError("estimated_state or state should appear in "
                             "the estimation results")
        metadata_filename = args.metadata_filename_pattern.format(est_res_number)
        metadata = configparser.ConfigParser()
        metadata.read(metadata_filename)
        estimation_params = dict(metadata["estimation_params"])
        if not "in_est_res_number" in estimation_params:
            exit_loop = True

    est_res_numbers_str = [str(est_res_number) for est_res_number in
                           est_res_numbers]
    fig = go.Figure()
    trace = go.Scatter(x=est_res_numbers_str, y=field_attrs)
    fig.add_trace(trace)

    fig.update_xaxes(
        title_text='Model ID',
        type='category',
        categoryorder='array',
        categoryarray=est_res_numbers_str,
        tickangle=-90  # Rotates labels 90 degrees vertically
    )

    fig.update_yaxes(title_text=field)

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    print("Saved {:s}".format(fig_filename_pattern.format("html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
