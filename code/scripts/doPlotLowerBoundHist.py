
import sys
import configparser
import pickle
import argparse

import svGPFA.plot.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        default=556223)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--estimated_model_filename_pattern",
                        help="saved estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--inferred_model_filename_pattern",
                        help="saved inferred model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_lower_bound_hist_vs_{{:s}}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
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
    lower_bound_hist = est_results["lower_bound_hist"]
    elapsed_time_hist = est_results["elapsed_time_hist"]


    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        lower_bound_hist=lower_bound_hist)
    fig.write_image(fig_filename_pattern.format("iteration", "png"))
    fig.write_html(fig_filename_pattern.format("iteration", "html"))

    print(f'Figure saved to {fig_filename_pattern.format("iteration", "html")}')

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        elapsed_time_hist=elapsed_time_hist, lower_bound_hist=lower_bound_hist)
    fig.write_image(fig_filename_pattern.format("elapsed_time", "png"))
    fig.write_html(fig_filename_pattern.format("elapsed_time", "html"))

    print(f'Figure saved to {fig_filename_pattern.format("elapsed_time", "html")}')

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
