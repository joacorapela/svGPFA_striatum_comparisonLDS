import sys
import pickle
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import plotly.graph_objects as go


def get_metric_matrix(metric, coupling_metrics, keys):
	metric_matrix = np.zeros(shape=(len(keys), len(keys)))
	for i in range(len(keys) - 1):
		for j in range(i+1, len(keys)):
			metric_matrix[i, j] = coupling_metrics[keys[i]][keys[j]][metric]
	return metric_matrix

def get_metric_plot(metric, coupling_metrics, keys):
    metric_matrix = get_metric_matrix(metric=metric,
                                      coupling_metrics=coupling_metrics,
                                      keys=keys)
    fig = go.Figure()
    trace = go.Heatmap(z=metric_matrix, x=keys, y=keys, colorbar={"title": metric})
    fig.add_trace(trace)

    return fig

def get_at_path(pytree, key):
    """Retrieves a sub-tree by dictionary key or list index."""
    if isinstance(key, (list, tuple)):
        val = pytree
        for k in key:
            val = val[k]
        return val
    return pytree[key]


def flatten_pytree_block_to_2d(block_pytree, outer_params, target_params):
    """
    Flattens a PyTree-of-PyTrees Hessian block into a 2D scalar matrix (N_outer, N_target).
    Works reliably across dictionaries, lists, and multi-dimensional parameter arrays.
    """
    outer_leaves = jax.tree_util.tree_leaves(outer_params)
    target_leaves = jax.tree_util.tree_leaves(target_params)

    outer_sizes = [x.size for x in outer_leaves]
    target_sizes = [x.size for x in target_leaves]

    # Map the inner target tree extraction per leaf of outer_params
    # block_pytree outer level matches outer_params leaf structure
    row_blocks = []
    
    # We iterate over outer leaves by unrolling leaves of outer_params
    # and extracting corresponding sub-trees from block_pytree
    flat_block_leaves = jax.tree_util.tree_leaves(block_pytree)
    
    num_target = len(target_leaves)
    num_outer = len(outer_leaves)

    for i in range(num_outer):
        col_blocks = []
        n_row = outer_sizes[i]
        
        for j in range(num_target):
            # Index into the flattened Hessian leaf list: (i * num_target + j)
            leaf_idx = i * num_target + j
            leaf_tensor = flat_block_leaves[leaf_idx]
            
            n_col = target_sizes[j]
            col_blocks.append(jnp.asarray(leaf_tensor).reshape(n_row, n_col))

        row_blocks.append(jnp.hstack(col_blocks))

    return jnp.vstack(row_blocks)


def get_sub_blocks_from_hessian_pytree(hessian, params, group_a_key, group_b_key=None):
    """Extracts 2D sub-matrices H_AA, H_BB, and H_AB directly from a PyTree Hessian."""
    
    params_A = get_at_path(params, group_a_key)
    H_A_outer = get_at_path(hessian, group_a_key)

    def extract_inner_target(outer_leaf_tree, target_key):
        return jax.tree_util.tree_map(
            lambda node: get_at_path(node, target_key),
            outer_leaf_tree,
            is_leaf=lambda node: isinstance(node, (dict, list, tuple)) and (
                (isinstance(node, dict) and target_key in node) or
                (isinstance(target_key, int) and isinstance(node, (list, tuple)) and len(node) > target_key)
            )
        )

    # 1. Extract Block (A, A)
    block_AA_pytree = extract_inner_target(H_A_outer, group_a_key)
    H_AA = flatten_pytree_block_to_2d(block_AA_pytree, params_A, params_A)

    if group_b_key is None:
        return H_AA, None, None

    params_B = get_at_path(params, group_b_key)
    H_B_outer = get_at_path(hessian, group_b_key)

    # 2. Extract Block (B, B)
    block_BB_pytree = extract_inner_target(H_B_outer, group_b_key)
    H_BB = flatten_pytree_block_to_2d(block_BB_pytree, params_B, params_B)

    # 3. Extract Cross-Block (A, B)
    block_AB_pytree = extract_inner_target(H_A_outer, group_b_key)
    H_AB = flatten_pytree_block_to_2d(block_AB_pytree, params_A, params_B)

    return H_AA, H_BB, H_AB

def get_coupling(H_pytree, group_a_key, group_b_key, group_a_tree, group_b_tree):
    """
    Calculates normalized coupling metrics (including Singular Value Ratio) 
    directly from a PyTree Hessian.
    """
    # 1. Extract 2D sub-blocks
    H_AA, H_BB, H_AB = get_sub_blocks_from_hessian_pytree(H_pytree, group_a_key, group_b_key)

    # -------------------------------------------------------------
    # METRIC A: Relative Frobenius Norm Ratio (Global Group Coupling)
    # -------------------------------------------------------------
    norm_AA_fro = jnp.linalg.norm(H_AA, ord='fro')
    norm_BB_fro = jnp.linalg.norm(H_BB, ord='fro')
    norm_AB_fro = jnp.linalg.norm(H_AB, ord='fro')
    
    frob_coupling = norm_AB_fro / (jnp.sqrt(norm_AA_fro * norm_BB_fro) + 1e-12)

    # -------------------------------------------------------------
    # METRIC B: Singular Value Ratio / Spectral Norm (Worst-Case Direction)
    # sigma_max(H_AB) / sqrt(sigma_max(H_AA) * sigma_max(H_BB))
    # -------------------------------------------------------------
    # Compute largest singular values (2-norms)
    sigma_max_AA = jnp.linalg.norm(H_AA, ord=2)
    sigma_max_BB = jnp.linalg.norm(H_BB, ord=2)
    sigma_max_AB = jnp.linalg.norm(H_AB, ord=2)

    singular_value_ratio = sigma_max_AB / (jnp.sqrt(sigma_max_AA * sigma_max_BB) + 1e-12)

    # -------------------------------------------------------------
    # METRIC 3: Max Element-wise Normalized Correlation
    # -------------------------------------------------------------
    diag_A = jnp.clip(jnp.diag(H_AA), a_min=1e-12)
    diag_B = jnp.clip(jnp.diag(H_BB), a_min=1e-12)
    outer_std = jnp.sqrt(jnp.outer(diag_A, diag_B))
    corr_matrix = jnp.abs(H_AB) / outer_std

    max_flat_idx = jnp.argmax(corr_matrix)
    flat_i, flat_j = jnp.unravel_index(max_flat_idx, corr_matrix.shape)
    max_corr_val = float(corr_matrix[flat_i, flat_j])

    # Map flat 1D indices back to N-D Tensor Coordinates
    def unflatten_index(flat_index, PyTree_structure):
        leaves = jax.tree_util.tree_leaves(PyTree_structure)
        shapes = [leaf.shape for leaf in leaves]
        sizes = [leaf.size for leaf in leaves]
        
        cumulative_sizes = jnp.cumsum(jnp.array([0] + sizes))
        leaf_idx = int(jnp.digitize(flat_index, cumulative_sizes) - 1)
        local_flat_idx = flat_index - int(cumulative_sizes[leaf_idx])
        
        nd_coord = jnp.unravel_index(local_flat_idx, shapes[leaf_idx])
        return leaf_idx, [int(c) for c in nd_coord]

    leaf_A, coord_A = unflatten_index(flat_i, group_a_tree)
    leaf_B, coord_B = unflatten_index(flat_j, group_b_tree)

    print("=== PyTree Hessian Coupling Analysis ===")
    print(f"Group A ({group_a_key}) Size       : {H_AA.shape[0]} elements")
    print(f"Group B ({group_b_key}) Size       : {H_BB.shape[0]} elements")
    print(f"Metric A (Frobenius Ratio)        : {float(frob_coupling):.4f}")
    print(f"Metric B (Singular Value Ratio)   : {float(singular_value_ratio):.4f}")
    print(f"Metric C (Max Element-wise Corr)  : {max_corr_val:.4f}")
    print(f"  └─ Peak Pair Location            : Leaf #{leaf_A} {coord_A} ({group_a_key}) <-> Leaf #{leaf_B} {coord_B} ({group_b_key})")

    return {
        "frob_coupling": float(frob_coupling),
        "singular_value_ratio": float(singular_value_ratio),
        "max_corr": max_corr_val,
        "max_pair_coords": ((leaf_A, coord_A), (leaf_B, coord_B)),
        "singular_values": {
            "sigma_max_AA": float(sigma_max_AA),
            "sigma_max_BB": float(sigma_max_BB),
            "sigma_max_AB": float(sigma_max_AB),
        }
    }
def get_coupling_metric(hessian, params, group_a_key, group_b_key):
    """Calculates normalized coupling metrics for N-D parameter groups directly from a PyTree Hessian."""
    
    # 1. Extract 2D sub-blocks directly using group keys
    H_AA, H_BB, H_AB = get_sub_blocks_from_hessian_pytree(
        hessian=hessian,
        params=params,
        group_a_key=group_a_key,
        group_b_key=group_b_key
    )
    
    # Extract parameter sub-trees internally
    group_a_params = params[group_a_key]
    group_b_params = params[group_b_key]

    # -------------------------------------------------------------
    # METRIC A: Relative Frobenius Norm Ratio (Global Group Coupling)
    # -------------------------------------------------------------
    norm_AA = jnp.linalg.norm(H_AA, ord='fro')
    norm_BB = jnp.linalg.norm(H_BB, ord='fro')
    norm_AB = jnp.linalg.norm(H_AB, ord='fro')

    frobenius_ratio = norm_AB / jnp.sqrt(norm_AA * norm_BB)

    # -------------------------------------------------------------
    # METRIC B: Singular Value Ratio / Spectral Norm (Worst-Case Direction)
    # sigma_max(H_AB) / sqrt(sigma_max(H_AA) * sigma_max(H_BB))
    # -------------------------------------------------------------
    # Compute largest singular values (2-norms)
    sigma_max_AA = jnp.linalg.norm(H_AA, ord=2)
    sigma_max_BB = jnp.linalg.norm(H_BB, ord=2)
    sigma_max_AB = jnp.linalg.norm(H_AB, ord=2)

    singular_value_ratio = sigma_max_AB / (jnp.sqrt(sigma_max_AA * sigma_max_BB) + 1e-12)

    # -------------------------------------------------------------
    # METRIC 3: Max Element-wise Normalized Correlation
    # -------------------------------------------------------------
    diag_A = jnp.clip(jnp.diag(H_AA), a_min=1e-12)
    diag_B = jnp.clip(jnp.diag(H_BB), a_min=1e-12)
    outer_std = jnp.sqrt(jnp.outer(diag_A, diag_B))

    corr_matrix = jnp.abs(H_AB) / outer_std

    max_flat_idx = jnp.argmax(corr_matrix)
    row_idx, col_idx = jnp.unravel_index(max_flat_idx, corr_matrix.shape)
    peak_val = corr_matrix[row_idx, col_idx]

    return {
        "frobenius_ratio": float(frobenius_ratio),
        "singular_value_ratio": float(singular_value_ratio),
        "singular_values": {
            "sigma_max_AA": float(sigma_max_AA),
            "sigma_max_BB": float(sigma_max_BB),
            "sigma_max_AB": float(sigma_max_AB),
        },
        "max_correlation": float(peak_val),
        "peak_pair_indices": (int(row_idx), int(col_idx)),
        "group_a_size": H_AA.shape[0],
        "group_b_size": H_BB.shape[0],
        "H_AA": H_AA,
        "H_BB": H_BB,
        "H_AB": H_AB,
    }

def get_coupling_metrics(hessian, params, verbose=True):
    keys = list(params.keys())
    coupling_metrics = dict()
    for i in range(len(keys)-1):
        group_a_key = keys[i]
        partial_coupling_metrics = dict()
        for j in range(i+1, len(keys)):
            group_b_key = keys[j]
            coupling = get_coupling_metric(
                hessian=hessian,
                params=params,
                group_a_key=group_a_key,
                group_b_key=group_b_key,
            )
            partial_coupling_metrics[group_b_key] = coupling

            if verbose:
                print(f"\n=== PyTree Hessian Coupling Analysis ===")
                print(f'Group A ({group_a_key}) Size : {coupling["H_AA"].shape[0]} scalar elements')
                print(f'Group B ({group_b_key}) Size : {coupling["H_BB"].shape[0]} scalar elements')
                print(f'Frobenius Coupling Ratio     : {coupling["frobenius_ratio"]:.4f}')
                print(f"Singular Value Ratio         : {coupling['singular_value_ratio']:.4f}")
                print(f'Max Element-wise Correlation : {coupling["max_correlation"]:.4f}')
                print(f' Peak Pair Matrix Location   : Row {coupling["peak_pair_indices"][0]}, Col {coupling["peak_pair_indices"][1]}')

        coupling_metrics[group_a_key] = partial_coupling_metrics

    return keys, coupling_metrics

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--est_res_number",
        help="estimation result number",
        type=int,
        default=79151056,
    )
    parser.add_argument(
        "--hessian_filename_pattern",
        help="model save filename pattern",
        type=str,
        default="../../results/EJT178_implant1/recording6_29-03-2022/{:08d}_hessian.pickle",
    )
    parser.add_argument(
        "--fig_filename_pattern",
        help="figure filename pattern",
        type=str,
        default="../../figures/EJT178_implant1/recording6_29-03-2022/{:08d}_coupling_{{:s}}.{{:s}}",
    )
    args = parser.parse_args()

    est_res_number = args.est_res_number
    hessian_filename = args.hessian_filename_pattern.format(est_res_number)
    fig_filename_pattern = args.fig_filename_pattern.format(est_res_number)

    with open(hessian_filename, "rb") as f:
        load_res = pickle.load(f)

    params = load_res["params"]
    hessian = load_res["hessian"]

    keys, coupling_metrics = get_coupling_metrics(hessian=hessian,
                                                  params=params)

    metric = "frobenius_ratio"
    frobenius_plot = get_metric_plot(metric=metric,
                                     coupling_metrics=coupling_metrics,
                                     keys=keys)
    frobenius_plot.write_html(fig_filename_pattern.format(metric, "html"))
    frobenius_plot.write_image(fig_filename_pattern.format(metric, "png"))

    print("Saved {:s}".format(fig_filename_pattern.format(metric, "html")))

    metric = "singular_value_ratio"
    frobenius_plot = get_metric_plot(metric=metric,
                                     coupling_metrics=coupling_metrics,
                                     keys=keys)
    frobenius_plot.write_html(fig_filename_pattern.format(metric, "html"))
    frobenius_plot.write_image(fig_filename_pattern.format(metric, "png"))

    print("Saved {:s}".format(fig_filename_pattern.format(metric, "html")))

    metric = "max_correlation"
    frobenius_plot = get_metric_plot(metric=metric,
                                     coupling_metrics=coupling_metrics,
                                     keys=keys)
    frobenius_plot.write_html(fig_filename_pattern.format(metric, "html"))
    frobenius_plot.write_image(fig_filename_pattern.format(metric, "png"))

    print("Saved {:s}".format(fig_filename_pattern.format(metric, "html")))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
