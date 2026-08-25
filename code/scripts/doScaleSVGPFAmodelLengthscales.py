import sys
import os.path
import random
import jax
import pickle
import argparse
import configparser

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_est_res_number", help="input estimation result number",
                        type=int,
                        # default=54368807)
                        default=33576128)
    parser.add_argument("--scale_factor", help="lengthscale scale factor",
                        type=float, default=1.0/50.0)
    parser.add_argument("--est_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_metaData.ini")
    parser.add_argument("--est_res_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_estimation_results.pickle")
    args = parser.parse_args()

    in_est_res_number = args.in_est_res_number
    scale_factor = args.scale_factor
    est_metadata_filename_pattern = args.est_metadata_filename_pattern
    est_res_filename_pattern = args.est_res_filename_pattern

    # save estimation initial conditions
    out_est_metadata = configparser.ConfigParser()
    out_est_metadata["script_info"] = {
        "name": __file__,
    }
    out_est_metadata["scale_factors"] = {
        "lengthscales": scale_factor,
    }
    out_est_metadata["estimation_params"] = {
        "in_est_res_number": in_est_res_number,
    }

    # build model_save_filename
    estPrefixUsed = True
    while estPrefixUsed:
        est_res_number = random.randint(0, 10**8)
        out_est_metadata_filename = \
            est_metadata_filename_pattern.format(est_res_number)
        if not os.path.exists(out_est_metadata_filename):
            estPrefixUsed = False
    out_est_res_filename = est_res_filename_pattern.format(est_res_number)

    with open(out_est_metadata_filename, "w") as f:
        out_est_metadata.write(f)
    print(f"Saved {out_est_metadata_filename}")

    in_est_res_filename = est_res_filename_pattern.format(in_est_res_number)
    with open(in_est_res_filename, "rb") as f:
        in_est_res = pickle.load(f)
    estimated_params = in_est_res["estimated_params"]
    kernels_params = estimated_params["kernels_params"]
    in_est_res["estimated_params"]["kernels_params"] = [kernel_param*scale_factor for kernel_param in kernels_params]

    breakpoint()

    with open(out_est_res_filename, "wb") as f:
        pickle.dump(in_est_res, f)
    print("Saved results to {:s}".format(out_est_res_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
