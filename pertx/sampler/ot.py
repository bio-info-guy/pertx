import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from functools import partial

# --- JAX Optimal Transport Engine ---
@partial(jax.jit, static_argnames=['top_k', 'epsilon'])
def _solve_ot_batch_with_cost(X_source, X_target, top_k=10, epsilon=0.1):
    """
    Returns: weights, local_indices, AND costs (squared euclidean distances)
    """
    # 1. Solve Sinkhorn
    geom = pointcloud.PointCloud(X_source, X_target, epsilon=epsilon)
    out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
    P = out.matrix
    
    # 2. Get Top-K Probabilities and Indices
    weights, local_indices = jax.lax.top_k(P, k=top_k)
    weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-10)
    
    # 3. Compute the Actual Distances for these Top-K pairs
    # X_source shape: (N, Dim) -> (N, 1, Dim)
    # X_target shape: (M, Dim) -> Gathered (N, K, Dim) using local_indices
    target_points = X_target[local_indices] 
    source_points = X_source[:, None, :]
    
    # Squared Euclidean Distance: sum((x - y)^2)
    dists = jnp.sum((source_points - target_points) ** 2, axis=-1)
    
    return weights, local_indices, dists

def compute_ot_for_subset(adata_subset, top_k=10, epsilon='auto', max_dist_sq="auto", pca_key='X_pca'):
    """
    Orchestrates the OT calculation for a specific subset of data.
    Returns: { cell_name: { pert_label: (target_names_list, weights_list) } }
    """
    ot_results = {}
    
    # Iterate by Cell Type to enforce biological constraint
    unique_celltypes = adata_subset.obs['celltype'].unique()
    
    for ctype in unique_celltypes:
        # Mask for this cell type
        mask_ctype = adata_subset.obs['celltype'] == ctype
        
        # Identify Sources (WT)
        mask_wt = mask_ctype & (adata_subset.obs['genotype'] == 'WT')
        if not np.any(mask_wt): continue 

        # Prepare Source Data
        X_wt = adata_subset.obsm[pca_key][mask_wt]
        wt_names = adata_subset.obs.index[mask_wt].tolist()
        
        # Convert to JAX array once per cell type (Source is constant)
        X_wt_jax = jnp.array(X_wt)

        # Iterate over Perturbations (Targets)
        available_perts = adata_subset.obs.loc[mask_ctype, 'genotype'].unique()
        available_perts = [p for p in available_perts if p != 'WT']
        
        for pert in available_perts:
            mask_pert = mask_ctype & (adata_subset.obs['genotype'] == pert)
            if mask_pert.sum() < top_k: continue # Skip if too few targets

            # Prepare Target Data
            X_pert = adata_subset.obsm[pca_key][mask_pert]
            target_names = adata_subset.obs.index[mask_pert].values 
            
            # --- DYNAMIC THRESHOLD CALCULATION ---
            # If "auto", we calculate the median distance for THIS SPECIFIC pair
            current_threshold = max_dist_sq if max_dist_sq is not None else np.inf
            if max_dist_sq == "auto":
                # We use the raw numpy arrays X_wt and X_pert
                current_threshold, suggested_eps = estimate_context_threshold(X_wt, X_pert)
            # -------------------------------------
            if epsilon == 'auto':
                epsilon = suggested_eps
            # Convert Target to JAX
            X_pert_jax = jnp.array(X_pert)
            
            # --- EXECUTE JAX ---
            try:
                weights, indices, dists = _solve_ot_batch_with_cost(
                    X_wt_jax, X_pert_jax, top_k=top_k, epsilon=epsilon
                )
                
                # Move to CPU
                weights_np = np.array(weights)
                indices_np = np.array(indices)
                dists_np = np.array(dists)
                
                # --- FORMAT RESULTS ---
                for i, wt_name in enumerate(wt_names):
                    # Filter based on distance of the best match
                    # dists_np[i, 0] is the squared distance to the #1 top-k match
                    if dists_np[i, 0] > current_threshold:
                        continue # Skip this cell, even its best match is too far

                    if wt_name not in ot_results: ot_results[wt_name] = {}
                    
                    # Map indices back to names
                    chosen_target_names = target_names[indices_np[i]]
                    ot_results[wt_name][pert] = (chosen_target_names, weights_np[i])
                    
            except Exception as e:
                print(f"OT Failed for {ctype} -> {pert}: {e}")
                
    return ot_results

def estimate_context_threshold(X_wt, X_pert, sample_size=1000):
    """
    Estimates a distance threshold specifically for this WT -> Pert pair.
    We compute the median distance between random pairs of (WT, Pert) cells.
    """
    from scipy.spatial.distance import cdist
    
    # Subsample if clouds are too large to speed up cdist
    if X_wt.shape[0] > sample_size:
        idx_wt = np.random.choice(X_wt.shape[0], sample_size, replace=False)
        X_wt_sub = X_wt[idx_wt]
    else:
        X_wt_sub = X_wt
        
    if X_pert.shape[0] > sample_size:
        idx_pert = np.random.choice(X_pert.shape[0], sample_size, replace=False)
        X_pert_sub = X_pert[idx_pert]
    else:
        X_pert_sub = X_pert
        
    # Calculate cross-distances (WT vs Pert)
    # cdist returns matrix of size (N_wt x N_pert)
    dists = cdist(X_wt_sub, X_pert_sub, metric='sqeuclidean')
    median_dist = np.median(dists)
    
    # Heuristic: Set epsilon to ~5-10% of the median squared distance
    # This ensures the exponent -dist/eps is roughly -10 to -20, preventing numerical collapse
    suggested_epsilon = median_dist * 0.1
    
    return median_dist, suggested_epsilon