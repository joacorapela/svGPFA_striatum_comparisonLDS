
import sys
import configparser
import pickle
import argparse
import numpy as np
import jax

import svGPFA.plot.plotUtilsPlotly

jax.config.update("jax_enable_x64", True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=54368807)
    parser.add_argument("--first_iteration_to_plot",
                        help="first iteration to plot", type=int,
                        default=0)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--estimated_model_filename_pattern",
                        help="saved estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    parser.add_argument("--inferred_model_filename_pattern",
                        help="saved inferred model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_lower_bound_hist_diff_vs_{{:s}}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    first_iteration_to_plot = args.first_iteration_to_plot
    inferred = args.inferred
    estimated_model_filename_pattern = args.estimated_model_filename_pattern
    inferred_model_filename_pattern = args.inferred_model_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    if inferred:
        model_filename = inferred_model_filename_pattern.format(est_res_number)
    else:
        model_filename = estimated_model_filename_pattern.format(est_res_number)

    with open(model_filename, "rb") as f:
        est_results = pickle.load(f)
    lower_bound_hist_diffs = np.diff(np.array(est_results["lower_bound_hist"]))
    elapsed_time_hist = est_results["elapsed_time_hist"]
    iterations = np.arange(len(lower_bound_hist_diffs))

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        iterations=iterations[first_iteration_to_plot:],
        lower_bound_hist=lower_bound_hist_diffs[first_iteration_to_plot:],
        ylabel="Lower Bound Difference")
    fig.write_image(fig_filename_pattern.format("iteration", "png"))
    fig.write_html(fig_filename_pattern.format("iteration", "html"))

    print(f'Figure saved to {fig_filename_pattern.format("iteration", "html")}')

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        elapsed_time_hist=elapsed_time_hist[(first_iteration_to_plot+1):],
        lower_bound_hist=lower_bound_hist_diffs[first_iteration_to_plot:],
        ylabel="Lower Bound Difference")
    fig.write_image(fig_filename_pattern.format("elapsed_time", "png"))
    fig.write_html(fig_filename_pattern.format("elapsed_time", "html"))

    print(f'Figure saved to {fig_filename_pattern.format("elapsed_time", "html")}')

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
