
import jax
import jax.numpy as jnp
import jax.flatten_util


def get_coordinated_curvatures(loss_fn, params, micro_batch_size=10):
    """
    Memory-safe Hessian diagonal calculation for svGPFA or heavy GP models.
    Uses tiny micro-batches (e.g. 10 parameters at a time) to prevent VRAM spikes.
    """
    # 1. Flatten svGPFA parameter tree into 1D
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    num_params = flat_params.shape[0]

    # 2. Define flat loss function
    def flat_loss(flat_p):
        return loss_fn(unravel_fn(flat_p))

    flat_grad_fn = jax.grad(flat_loss)

    # 3. Compute single diagonal element
    def single_diagonal_element(i):
        e_i = jnp.zeros(num_params).at[i].set(1.0)
        _, h_i = jax.jvp(flat_grad_fn, (flat_params,), (e_i,))
        return h_i[i]

    # 4. Vmap ONLY across a tiny micro-batch (e.g. 10 parameters at once)
    @jax.jit
    def process_micro_batch(batch_indices):
        return jax.vmap(single_diagonal_element)(batch_indices)

    # 5. Loop sequentially across all 160,000 parameters
    flat_diag_list = []

    print(f"Total parameters: {num_params}. Processing in micro-batches of size {micro_batch_size} ...")

    for start_idx in range(0, num_params, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, num_params)
        batch_indices = jnp.arange(start_idx, end_idx)

        # Compute tiny micro-batch
        batch_diag = process_micro_batch(batch_indices)
        batch_diag.block_until_ready()

        flat_diag_list.append(batch_diag)

        # Print progress every 10,000 parameters
        if (start_idx + micro_batch_size) % 10000 < micro_batch_size:
            print(f"  Processed {min(start_idx + micro_batch_size, num_params)} / {num_params} parameters...")

    # 6. Reconstruct PyTree matching `params` structure
    flat_diag = jnp.concatenate(flat_diag_list)
    answer = unravel_fn(flat_diag)
    return answer


