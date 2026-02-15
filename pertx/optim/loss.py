# Modified from scGPT
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    if not mask.any():
        mask = torch.ones_like(input, dtype=torch.bool)
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    
    if mask is None:
        return F.mse_loss(input, target, reduction="mean")
    if not mask.any():
        mask = torch.ones_like(input, dtype=torch.bool)
    mask = mask.float() 
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    
    if mask is None:
        bernoulli = torch.distributions.Bernoulli(probs=input)
        masked_log_probs = bernoulli.log_prob((target > 0).float())
        return -masked_log_probs.mean()
    if not mask.any():
        mask = torch.ones_like(input, dtype=torch.bool)
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


    
    
def perturb_embedding_loss(
    input_emb: torch.Tensor,
    input_to_pert_emb: torch.Tensor,
    pert_emb: torch.Tensor,
    pert_to_input_emb: torch.Tensor,
    lambda_fwd: float = 1.0,
    lambda_rev: float = 1.0
) -> torch.Tensor:
    """
    Calculates the composite loss for the virtual perturbation model.

    This loss combines three components:
    1. Reconstruction Loss (MSE): How well the decoded expression matches the true one.
    2. Forward Consistency Loss (Cosine Distance): Enforces that the predicted perturbed
       embedding is close to the true perturbed embedding.
    3. Reverse Consistency Loss (Cosine Distance): Enforces cycle consistency, ensuring
       the reverse-perturbed embedding is close to the original input embedding.

    Args:
        decoded_expression (torch.Tensor): The final output from the decoder (predicted expression).
        true_expression (torch.Tensor): The ground truth expression of the sampled perturbed cell.
        input_emb (torch.Tensor): The embedding of the original input cell.
        pert_to_input_emb (torch.Tensor): The result of reverse-perturbing the perturbed cell's embedding.
        input_to_pert_emb (torch.Tensor): The result of perturbing the input cell's embedding (the predicted perturbed embedding).
        pert_emb (torch.Tensor): The true embedding of the sampled perturbed cell.
        lambda_fwd (float): The weight for the forward consistency loss.
        lambda_rev (float): The weight for the reverse consistency loss.

    Returns:
        torch.Tensor: A single scalar value representing the total loss.
    """
    #mask = input_labels != pert_labels
    #mask = mask.unsqueeze(1)

    # Forward Consistency Loss (L_fwd_consistency)
    # Cosine distance = 1 - Cosine Similarity.
    # We want to maximize similarity, which is equivalent to minimizing distance.
    # The '.mean()' aggregates the loss across the batch.
    #similarity_fwd = F.Cosine(input_to_pert_emb, pert_emb)
    #loss_fwd_consistency = (1 - similarity_fwd).mean()
    #loss_fwd_consistency = F.relu(F.mse_loss(input_to_pert_emb, pert_emb) -  F.mse_loss(input_to_pert_emb*mask, input_emb*mask) + 0.5)
    loss_fwd_consistency = F.mse_loss(input_to_pert_emb, pert_emb)# -  F.mse_loss(input_to_pert_emb*mask, input_emb*mask) + 0.5)
    #  Reverse Consistency Loss (L_rev_consistency)
    # Similar to the forward loss, this ensures the reverse transformation is valid.
    #similarity_rev = F.mse_loss(pert_to_input_emb, input_emb)
    #loss_rev_consistency = F.relu(F.mse_loss(pert_to_input_emb, input_emb) - F.mse_loss(pert_to_input_emb*mask, pert_emb*mask) + 0.5)
    loss_rev_consistency = F.mse_loss(pert_to_input_emb, input_emb)# - F.mse_loss(pert_to_input_emb*mask, pert_emb*mask) + 0.5)

    # 4. Combine the losses
    # The total loss is a weighted sum of the three components.
    total_loss = lambda_fwd * loss_fwd_consistency + lambda_rev * loss_rev_consistency
    
    # You can optionally return the individual components for monitoring during training
    # return total_loss, loss_recon, loss_fwd_consistency, loss_rev_consistency
    
    return total_loss


    
def SUPCON_loss(features, labels=None, mask=None, contrast_mode = 'all', temperature = 0.07, base_temperature = 0.5, normalize_logits = False):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    dim = features.shape[-1]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature*dim)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    logits = logits * logits_mask # exclude self logit from normalization if we normalize
    if normalize_logits:
        # Normalize the logits for each anchor
        norms = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
        logits = torch.div(logits, norms)
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point. 
    # Edge case e.g.:- 
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan] 
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

# wrapper function to calculate cce loss on pertTF outputs contrastive dictionary
def cce_loss(contrastive_dict, input_labels, pert_labels, logit_norm = False, positions = None):
    if positions == None:
        positions = torch.ones(input_labels.size(0), dtype=torch.bool, device=input_labels.device)
    loss_cce = 0
    if len(contrastive_dict) == 4:
        loss_cce += perturb_embedding_loss(
            contrastive_dict['orig_emb0'],
            contrastive_dict['next_emb0'],
            contrastive_dict['next_emb1'],
            contrastive_dict['orig_emb1'],
            lambda_fwd=5,
            lambda_rev=5
        ) 
    contr_keys = list(contrastive_dict.keys())
    emb_list = [contrastive_dict[k] if 'orig' in k else contrastive_dict[k][positions] for k in contr_keys]
    lab_list = [input_labels if 'orig' in k else pert_labels[positions] for k in contr_keys]
    loss_cce += SUPCON_loss(
        features = torch.concat(emb_list, dim = 0).unsqueeze(1), 
        labels = torch.concat(lab_list),
        normalize_logits = logit_norm
    )
    return loss_cce

"""
-----------------------------------------
Optional Losses Implemented but not used
-----------------------------------------
"""

def semi_masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha = 0.7
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss_mask = F.mse_loss(input * mask, target * mask, reduction="sum") / mask.sum()
    loss_other = F.mse_loss(input * (1 - mask), target * (1 - mask), reduction="sum") / (1- mask).sum()
    loss = loss_other*(1-alpha) + loss_mask*alpha
    return loss 


def criterion_semi_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha = 0.7
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    log_probs = bernoulli.log_prob((target > 0).float())
    masked_log_probs_mask = log_probs * mask / mask.sum()
    masked_log_probs_other = log_probs * (1- mask) / (1 - mask).sum()
    masked_log_probs = masked_log_probs_other*(1-alpha) + masked_log_probs_mask*alpha
    return -masked_log_probs.sum()


def semi_masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor, alpha = 0.7
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    #loss_mask = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    loss = torch.abs(input- target) / (target + 1e-6)
    loss[mask] = loss[mask]*alpha
    loss[~mask] = loss[~mask]*(1-alpha)
    #loss_other = torch.abs(input[~mask] - target[~mask]) / (target[~mask] + 1e-6)
    #loss = loss_other*(1-alpha) + loss_mask*alpha
    return loss.mean()





def l1_loss_flexible(
    v_head: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    p_head: Optional[torch.Tensor] = None,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Computes a flexible L1 loss with weighted masking.

    If p_head is provided, it computes the unified loss where the final prediction
    is the product of the probability and value heads (y_pred = p_head * v_head).

    If p_head is None, it computes a standard L1 loss directly on the value
    head (y_pred = v_head).

    Args:
        v_head (torch.Tensor): The output of the continuous value head.
        target (torch.Tensor): The ground truth sparse vector.
        mask (torch.Tensor): The binary mask tensor. 1 for primary loss positions.
        p_head (Optional[torch.Tensor]): The optional output of the probability
                                           head (after sigmoid, in [0, 1]).
                                           Defaults to None.
        alpha (float): The weight for the loss on the masked positions.

    Returns:
        torch.Tensor: The final computed loss value.
    """
    # Step 1: Create the prediction based on whether p_head is provided
    if p_head is not None:
        # Unified model: prediction is the gated value
        y_pred = p_head * v_head
    else:
        # Standard model: prediction is just the value
        y_pred = v_head

    # --- The rest of the logic remains the same ---

    # Ensure mask is float for calculations
    mask = mask.float()

    # Step 2: Calculate the per-element absolute error
    abs_error = torch.abs(y_pred - target)

    # Step 3: Calculate the mean loss for the masked and unmasked parts separately
    # Add a small epsilon (1e-8) to the denominator to prevent division by zero
    sum_mask = mask.sum()
    loss_mask = (abs_error * mask).sum() / (sum_mask + 1e-8)

    sum_other = (1 - mask).sum()
    loss_other = (abs_error * (1 - mask)).sum() / (sum_other + 1e-8)

    # Step 4: Combine the losses using the alpha weight
    loss = loss_mask * alpha + loss_other * (1 - alpha)
    
    # Handle cases where one of the masks is empty
    if sum_mask == 0 and sum_other > 0:
        loss = loss_other
    elif sum_other == 0 and sum_mask > 0:
        loss = loss_mask
        
    return loss


class MaskedNBZINBLoss(nn.Module):
    def __init__(self, zero_inflation: bool = True, eps: float = 1e-6):
        """
        Args:
            zero_inflation (bool): If True, uses ZINB loss. If False, uses standard NB loss.
            eps (float): Numerical stability term.
        """
        super().__init__()
        self.zero_inflation = zero_inflation
        self.eps = eps

    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        dispersion: torch.Tensor, 
        pi: torch.Tensor = None, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input (Tensor): Predicted mean (mu) from the model (must be positive).
            target (Tensor): True raw counts (integers).
            dispersion (Tensor): Predicted dispersion (theta).
            pi (Tensor, optional): Predicted non-zero probability. Required if zero_inflation=True.
            mask (Tensor, optional): Boolean mask for valid positions.
        """
        # 1. Input Validation & Stability
        mean = input#['pred_value'] # Renaming for clarity
        target = torch.round(target)
        target = torch.clamp(target, min=0)
        #dispersion = input['dispersion']
        #pi = input['zero_probs']
        if self.zero_inflation and pi is None:
            raise ValueError("If zero_inflation is True, 'pi' (dropout prob) must be provided.")
        
        # Clamp for numerical stability
        mean = torch.clamp(mean, min=self.eps, max=1e5)
        dispersion = torch.clamp(dispersion, min=1e-4, max=1e4)
        if self.zero_inflation:
            pi = 1 - pi # technically pi is non zero probability in mvc output, convert to dropout prob
            pi = torch.clamp(pi, min=1e-4, max=0.999)

        # 2. Calculate Standard Negative Binomial (NB) Log Likelihood
        # Formula: LL(y; mu, theta)
        # t1 = lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
        t1 = torch.lgamma(target + dispersion) - torch.lgamma(dispersion) - torch.lgamma(target + 1)
        # t2 = theta * log(theta) + y * log(mean)
        t2 = dispersion * torch.log(dispersion) + target * torch.log(mean)
        # t3 = (theta + y) * log(theta + mean)
        t3 = (dispersion + target) * torch.log(dispersion + mean)
        
        log_nb = t1 + t2 - t3

        # 3. Handle Zero-Inflation (or skip it)
        if self.zero_inflation:
            final_log_prob = torch.zeros_like(log_nb)
            
            # --- Branch 1: Target > 0 ---
            # For non-zeros, the probability is just (1-pi) * NB(y)
            # We use a mask to compute this ONLY where target > 0
            nonzero_mask = (target > 0)
            
            if nonzero_mask.any():
                log_zinb_nonzero = (
                    torch.log(1 - pi[nonzero_mask]) + 
                    log_nb[nonzero_mask]
                )
                final_log_prob[nonzero_mask] = log_zinb_nonzero
            # --- Branch 2: Target == 0 ---
            # For zeros, we have a mixture: pi + (1-pi) * NB(0)
            zero_mask = ~nonzero_mask
            
            if zero_mask.any():
                # Extract values for zero positions only to save computation
                pi_zero = pi[zero_mask]
                theta_zero = dispersion[zero_mask]
                mean_zero = mean[zero_mask]
                
                # NB(0) log probability: theta * (log(theta) - log(theta + mean))
                log_nb_zero_val = theta_zero * (torch.log(theta_zero) - torch.log(theta_zero + mean_zero))
                
                # Robust LogAddExp: log(exp(a) + exp(b))
                # log( pi + (1-pi)*NB(0) )
                log_zinb_zero = torch.logaddexp(
                    torch.log(pi_zero), 
                    torch.log(1 - pi_zero) + log_nb_zero_val
                )
                final_log_prob[zero_mask] = log_zinb_zero
        else:
            # --- Standard NB Logic ---
            # Just use the raw NB log probability
            final_log_prob = log_nb

        # 4. Apply Masking (Matches your MSE logic)
        # We want Negative Log Likelihood, so flip sign
        loss_elementwise = -final_log_prob

        if mask is None:
            return loss_elementwise.mean()
        
        # If mask is empty/all-false, handle gracefully (or treat all as True like your code)
        if not mask.any():
            mask = torch.ones_like(loss_elementwise, dtype=torch.bool)
            
        mask = mask.float()
        
        # Sum masked loss and divide by number of masked elements
        masked_loss_sum = (loss_elementwise * mask).sum()
        mask_sum = mask.sum()
        
        return masked_loss_sum / mask_sum, final_log_prob


def all_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    Calculates the triplet loss for a batch of embeddings using a "batch-all" strategy.

    This method considers all valid anchor-positive-negative triplets within the batch.
    A triplet is valid if the anchor and positive have the same label, and the anchor
    and negative have different labels. The loss is then averaged over all triplets
    that have a positive loss value.

    Args:
        embeddings (torch.Tensor): The batch of embeddings (shape: [batch_size, emb_dim]).
        labels (torch.Tensor): The labels for each embedding (shape: [batch_size]).
        margin (float): The desired margin between positive and negative distances.

    Returns:
        torch.Tensor: A single scalar value for the mean triplet loss.
    """
    # Calculate pairwise squared L2 distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2).pow(2)

    # Create masks to identify positive and negative pairs
    mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0))

    
    mask_positive.fill_diagonal_(False)
    
    mask_negative = ~mask_positive
    mask_negative.fill_diagonal_(False)
    hardest_negative_dist = (pairwise_dist + 1e8 * (~mask_negative)).min(dim=1)[0]
    # --- Batch-All Triplet Mining ---
    # For each anchor, we want to consider all positive and all negative pairs.
    # We can use broadcasting to compute the loss for all possible triplets.
    
    # Reshape distances for broadcasting:
    # anchor_positive_dist[i, j] = distance(i, j)
    # anchor_negative_dist[i, k] = distance(i, k)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)  # Shape: (batch, batch, 1)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)  # Shape: (batch, 1, batch)

    # Calculate the loss for all possible triplets (i, j, k)
    # triplet_loss[i, j, k] = D(i, j) - D(i, k) + margin
    triplet_loss = (anchor_positive_dist - anchor_negative_dist)/hardest_negative_dist.mean() + margin

    # Create a mask for valid triplets. A triplet (i, j, k) is valid if
    # (i, j) is a positive pair and (i, k) is a negative pair.
    mask_valid_triplets = mask_positive.unsqueeze(2) & mask_negative.unsqueeze(1)
    
    # Apply the mask to keep only the loss for valid triplets
    # Set the loss for invalid triplets to 0
    triplet_loss = triplet_loss * mask_valid_triplets
    
    # Remove negative losses (as per the max(0, loss) formulation)
    triplet_loss = F.relu(triplet_loss)

    # Count the number of triplets with positive loss
    num_positive_triplets = (triplet_loss > 1e-16).float().sum()
    
    # Calculate the mean loss over the positive triplets.
    # If there are no positive triplets, the loss is 0.
    if num_positive_triplets > 0:
        loss = triplet_loss.sum() / num_positive_triplets
    else:
        loss = torch.tensor(0.0, device=embeddings.device)

    return loss


def hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    Calculates the triplet loss for a batch of embeddings using online hard triplet mining.

    For each anchor in the batch, it finds the hardest positive (most distant sample
    with the same label) and the hardest negative (closest sample with a different
    label) and computes the loss.

    Args:
        embeddings (torch.Tensor): The batch of embeddings (shape: [batch_size, emb_dim]).
        labels (torch.Tensor): The labels for each embedding (shape: [batch_size]).
        margin (float): The desired margin between positive and negative distances.

    Returns:
        torch.Tensor: A single scalar value for the mean triplet loss.
    """
    # Calculate pairwise squared L2 distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2).pow(2)

    # Create masks to identify positive and negative pairs
    # mask_positive[i, j] is True if sample i and j have the same label
    mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0))
    # We need to ignore the distance of a sample to itself (diagonal)
    mask_positive.fill_diagonal_(False)
    
    # mask_negative[i, j] is True if sample i and j have different labels
    mask_negative = ~mask_positive
    mask_negative.fill_diagonal_(False)

    # --- Hard Triplet Mining ---
    # For each anchor, find the hardest positive (max distance)
    # Add a large negative value to non-positive pairs to ensure they aren't chosen
    hardest_positive_dist = (pairwise_dist + -1e8 * (~mask_positive)).max(dim=1)[0]

    # For each anchor, find the hardest negative (min distance)
    # Add a large positive value to non-negative pairs to ensure they aren't chosen
    hardest_negative_dist = (pairwise_dist + 1e8 * (~mask_negative)).min(dim=1)[0]
    
    # Calculate triplet loss for each sample in the batch
    # loss = max(0, D(anchor, positive) - D(anchor, negative) + margin)
    loss = F.relu((hardest_positive_dist - hardest_negative_dist)/ hardest_negative_dist.mean() + margin)

    return loss.mean()

import torch.nn.functional as F

def compute_hard_negative_link_loss(embeddings, master_edges, batch_size=4096, margin=0.1, num_neg_candidates=20):
    """
    Computes Margin Loss with Hard Negative Mining.
    
    num_neg_candidates (k): How many random negatives to test for each positive. 
                            Higher k = Harder negatives = Stronger signal.
    """
    device = embeddings.device
    num_nodes = embeddings.size(0)
    embedding_dim = embeddings.size(1)

    # --- 1. Normalize for Cosine Similarity ---
    # (Optional but recommended for stability)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # --- 2. Sample Positive Edges ---
    perm = torch.randint(0, master_edges.size(1), (batch_size,), device='cpu')
    batch_pos_edges = master_edges[:, perm].to(device)
    
    pos_src = embeddings[batch_pos_edges[0]] # [Batch, Dim]
    pos_tgt = embeddings[batch_pos_edges[1]] # [Batch, Dim]
    
    # Positive Scores (Cosine Sim)
    pos_scores = (pos_src * pos_tgt).sum(dim=-1) # [Batch]

    # --- 3. Hard Negative Mining ---
    # We want to find negatives that are deceptively close to 'pos_src'.
    
    # A. Generate 'k' random negative indices for EACH item in the batch
    # Shape: [Batch, k]
    neg_tgt_idx = torch.randint(0, num_nodes, (batch_size, num_neg_candidates), device=device)
    
    # B. Retrieve embeddings for all candidates
    # Shape: [Batch, k, Dim]
    neg_tgt_candidates = embeddings[neg_tgt_idx] 
    
    # C. Prepare source embeddings for broadcasting
    # Reshape pos_src from [Batch, Dim] -> [Batch, 1, Dim]
    pos_src_expanded = pos_src.unsqueeze(1)
    
    # D. Calculate scores for ALL candidates
    # (Batch, 1, Dim) * (Batch, k, Dim) -> Sum over Dim -> (Batch, k)
    all_neg_scores = (pos_src_expanded * neg_tgt_candidates).sum(dim=-1)
    
    # E. Pick the "Hardest" Negative (The one with the Highest Score)
    # We take the max along the candidate dimension (dim=1)
    hard_neg_scores, _ = all_neg_scores.max(dim=1) # [Batch]

    # --- 4. Compute Margin Loss ---
    # We want: pos_scores > hard_neg_scores + margin
    current_margin = margin - pos_scores + hard_neg_scores
    loss = torch.clamp(current_margin, min=0).mean()
    
    return loss

# --- 2. THE LOSS FUNCTION (Efficient) ---
def compute_margin_link_loss(embeddings, master_edges, batch_size=4096, margin=0.2):
    """
    Uses Margin Ranking Loss.
    Goal: Score(Real_Edge) > Score(Fake_Edge) + Margin
    """
    device = embeddings.device
    num_nodes = embeddings.size(0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    #scale_factor = 1.0 / (embeddings.size(-1) ** 0.5)
    
    # 1. Sample Positive Edges (Real)
    perm = torch.randint(0, master_edges.size(1), (batch_size,), device='cpu')
    batch_pos_edges = master_edges[:, perm].to(device)
    
    pos_src = embeddings[batch_pos_edges[0]]
    pos_tgt = embeddings[batch_pos_edges[1]]
    
    # Calculate Similarity (Dot Product)
    # Range: (-inf, +inf)
    pos_scores = (pos_src * pos_tgt).sum(dim=-1)#*scale_factor
    
    # 2. Sample Negative Edges (Fake)
    # We sample 1 negative for every 1 positive (1:1 Ratio)
    neg_src_idx = torch.randint(0, num_nodes, (batch_size,), device=device)
    neg_tgt_idx = torch.randint(0, num_nodes, (batch_size,), device=device)
    
    neg_src = embeddings[neg_src_idx]
    neg_tgt = embeddings[neg_tgt_idx]
    
    neg_scores = (neg_src * neg_tgt).sum(dim=-1)#*scale_factor
    
    # 3. Margin Loss Calculation
    # We want: pos_scores > neg_scores + margin
    # Equivalent to: margin - pos_scores + neg_scores < 0
    # Loss = max(0, margin - pos + neg)
    
    loss = torch.clamp(margin - pos_scores + neg_scores, min=0).mean()
    
    return loss


# --- 1. Loss Functions ---

def loss_zig(mu, theta, pi, target, eps=1e-6):
    """Zero-Inflated Gaussian Loss (Log Space)."""
    sigma = theta
    zero_mask = (target <= eps)
    pos_mask = (target > eps)
    
    loss = torch.zeros_like(target)
    
    # Gate Loss
    if zero_mask.any():
        loss[zero_mask] = -torch.log(pi[zero_mask])
    
    # Gaussian Loss
    if pos_mask.any():
        y = target[pos_mask]
        mu_p = mu[pos_mask]
        sigma_p = sigma[pos_mask]
        gaussian_nll = (
            torch.log(sigma_p) + 
            0.5 * 1.837877 + 
            0.5 * ((y - mu_p) / sigma_p)**2
        )
        loss[pos_mask] = -torch.log(1 - pi[pos_mask]) + gaussian_nll
    return loss

def loss_nb(mu, theta, pi, target, eps=1e-6):
    """Standard Negative Binomial Loss."""
    t1 = torch.lgamma(target + theta) - torch.lgamma(theta) - torch.lgamma(target + 1)
    t2 = theta * torch.log(theta) + target * torch.log(mu)
    t3 = (theta + target) * torch.log(theta + mu)
    return -(t1 + t2 - t3)

def loss_zinb(mu, theta, pi, target, eps=1e-6):
    """Zero-Inflated NB Loss."""
    nb_nll = loss_nb(mu, theta, pi, target, eps) # Get positive NLL
    loss = torch.zeros_like(target)
    
    # Non-Zero Targets
    nonzero_mask = (target > 0)
    if nonzero_mask.any():
        loss[nonzero_mask] = -torch.log(1 - pi[nonzero_mask]) + nb_nll[nonzero_mask]
        
    # Zero Targets
    zero_mask = ~nonzero_mask
    if zero_mask.any():
        pi_z, theta_z, mu_z = pi[zero_mask], theta[zero_mask], mu[zero_mask]
        log_nb_0 = theta_z * (torch.log(theta_z) - torch.log(theta_z + mu_z))
        
        loss[zero_mask] = -torch.logaddexp(
            torch.log(pi_z), 
            torch.log(1 - pi_z) + log_nb_0
        )
    return loss

def loss_hnb(mu, theta, pi, target, eps=1e-6):
    """Hurdle NB Loss."""
    loss = torch.zeros_like(target)
    zero_mask = (target == 0)
    pos_mask = (target > 0)

    # Gate Loss (BCE)
    with torch.autocast(device_type='cuda', enabled=False):
        loss += F.binary_cross_entropy(pi.float(), zero_mask.float(), reduction='none')

    # Truncated NB Loss
    if pos_mask.any():
        nb_nll = loss_nb(mu, theta, pi, target, eps)
        
        theta_p, mu_p = theta[pos_mask], mu[pos_mask]
        log_prob_zero = theta_p * (torch.log(theta_p) - torch.log(theta_p + mu_p))
        truncation = torch.log(-torch.expm1(log_prob_zero) + eps)
        
        loss[pos_mask] += (nb_nll[pos_mask] + truncation) # nb_nll is already positive
    return loss

def loss_pois(mu, theta, pi, target, eps=1e-6):
    """Standard Poisson."""
    return F.poisson_nll_loss(mu, target, log_input=False, full=False, reduction='none')

def loss_zipois(mu, theta, pi, target, eps=1e-6):
    """Zero-Inflated Poisson."""
    loss = torch.zeros_like(target)
    nonzero_mask = (target > 0)
    
    # Non-Zero
    if nonzero_mask.any():
        pois_nll = F.poisson_nll_loss(mu[nonzero_mask], target[nonzero_mask], 
                                      log_input=False, full=False, reduction='none')
        loss[nonzero_mask] = -torch.log(1 - pi[nonzero_mask]) + pois_nll
        
    # Zero
    zero_mask = ~nonzero_mask
    if zero_mask.any():
        pi_z, mu_z = pi[zero_mask], mu[zero_mask]
        # Pois(0) = exp(-mu) -> log(Pois(0)) = -mu
        loss[zero_mask] = -torch.logaddexp(
            torch.log(pi_z),
            torch.log(1 - pi_z) - mu_z
        )
    return loss

# --- 2. Loss Registry ---

LOSS_REGISTRY = {
    'zig': loss_zig,
    'nb': loss_nb,
    'zinb': loss_zinb,
    'hnb': loss_hnb,
    'pois': loss_pois,
    'zipois': loss_zipois
}

class GenerativeExpressionLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, 
        outputs: dict, 
        target: torch.Tensor, 
        mask: torch.Tensor = None,
        scale_factor: torch.Tensor = None,
        positions: torch.Tensor = None
    ) -> torch.Tensor:
        
        # control which samples contribute to loss
        if positions is None:
            positions = torch.ones(target.size(0), dtype=torch.bool, device=target.device)
        # 1. Handle Mask Defaults
        if mask is None or not mask.any():
            mask = torch.ones_like(outputs['pred'], dtype=torch.bool)
        
        
        # 2. Check Distribution Type
        dist_type = outputs.get('distribution', None)
        if dist_type is None:
            # Fallback to MSE
            # Note: Assuming masked_mse_loss handles masking internally
            return masked_mse_loss(outputs['pred'], target, mask) 
        
        if dist_type not in LOSS_REGISTRY:
             raise ValueError(f"Distribution '{dist_type}' not supported.")

        mask = mask[positions]
        # --- PRE-PROCESSING (While shapes are still 2D) ---
        
        # Un-normalize Targets (Log-Norm -> Counts)
        # We do this BEFORE masking so that (Batch, 1) scale_factor broadcasts 
        # correctly to (Batch, Genes)
        if dist_type != 'zig':
            if scale_factor is None:
                raise ValueError(f"{dist_type} requires scale_factor.")
            
            # NOTE: Check math here. 
            # raw count = expm1(log_norm)/scale_factor
            target = torch.expm1(target)/scale_factor
            target = torch.round(target).clamp(min=0)

        # --- MASKING (Flatten to 1D) ---
        # Now we flatten everything to only valid elements.
        # This prevents invalid params (e.g. negative theta in padding) from crashing loss.
        
        mu = outputs['pred'][positions][mask]
        target = target[positions][mask]
        
        param2 = outputs.get('param2')
        if param2 is not None:
            if param2.dim() == 1:
                param2 = param2.expand(outputs['pred'].shape[0], -1)
            param2 = param2[positions][mask]
            # Stability Clamp (Important for Theta/Sigma)
            param2 = torch.clamp(param2, min=1e-4, max=1e4)

        pi = outputs.get('zero_probs')
        if pi is not None:
            # Convert Prob(Non-Zero) -> Prob(Dropout)
            pi = 1.0 - pi 
            pi = pi[positions][mask]
            # Stability Clamp (Important for BCE/Log)
            pi = torch.clamp(pi, min=1e-4, max=0.999)

        # Clamp Mu
        mu = torch.clamp(mu, min=self.eps, max=1e6)

        # --- DISPATCH ---
        # Now all inputs are 1D vectors of valid elements.
        loss_func = LOSS_REGISTRY[dist_type]
        loss_elementwise = loss_func(mu, param2, pi, target, self.eps)

        # --- REDUCTION ---
        # Since we already filtered by mask, we just take the mean.
        return loss_elementwise.mean()

