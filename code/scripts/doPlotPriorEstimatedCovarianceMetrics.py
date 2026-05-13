
import sys
import configparser
import numpy as np
import pickle
import argparse
import plotly.graph_objects as go

import svGPFA.utils.miscUtils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int,
                        # default=66612199)
                        # default=38426992)
                        # default=99226606)
                        # default=99749566)
                        default=54368807)
                        # default=42976344)
                        # default=34655634)
                        # default=22746506)
                        # default=92418550)
                        # default=25058234)
                        # default=6092425)
                        # default=1283092)
                        # default=57339587)
                        # default=33576128)
                        # default=74463115)
                        # default=71504301)
    parser.add_argument("--inferred",
                        help="variables were inferred and not estimated",
                        action="store_true")
    parser.add_argument("--estimated_model_filename_pattern",
                        help="estimated model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimatedModel.pickle")
    parser.add_argument("--inferred_model_filename_pattern",
                        help="inferred model filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inferredModel.pickle")
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern", type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_inference_metaData.ini")
                        # default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_prior_cov_{{:s}}.{{:s}}")
    args = parser.parse_args()

    est_res_number = args.est_res_number
    inferred = args.inferred
    estimated_model_filename = args.estimated_model_filename_pattern.format(est_res_number)
    inferred_model_filename = args.inferred_model_filename_pattern.format(est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    if inferred:
        model_filename = inferred_model_filename
    else:
        model_filename = estimated_model_filename

    with open(model_filename, "rb") as f:
        est_results = pickle.load(f)
    estimated_params = est_results["estimated_params"]
    vChol = estimated_params["variational_chol_vecs"]
    K = vChol.shape[0]
    R = vChol.shape[1]

    vCov = svGPFA.utils.miscUtils.buildCovsFromCholVecs(vChol)
    dets = [[np.linalg.det(vCov[k, r, :, :]) for r in range(R)] for k in range(K)]
    conds = [[np.linalg.cond(vCov[k, r, :, :]) for r in range(R)] for k in range(K)]

    fig = go.Figure()
    for k in range(K):
        trace = go.Scatter(x=np.arange(R), y=dets[k], name=f"latent {k}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Trial")
    fig.update_yaxes(title="Determinant")
    fig.write_image(fig_filename_pattern.format("det", "png"))
    fig.write_html(fig_filename_pattern.format("det", "html"))

    print(f'Figure saved to {fig_filename_pattern.format("det", "html")}')

    fig = go.Figure()
    for k in range(K):
        trace = go.Scatter(x=np.arange(R), y=conds[k], name=f"latent {k}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Trial")
    fig.update_yaxes(title="Condition Number")
    fig.write_image(fig_filename_pattern.format("cond", "png"))
    fig.write_html(fig_filename_pattern.format("cond", "html"))

    print(f'Figure saved to {fig_filename_pattern.format("cond", "html")}')

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
