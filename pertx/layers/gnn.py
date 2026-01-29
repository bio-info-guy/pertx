from typing import Dict, Mapping, Optional, Tuple, Any, Union, List, Dict, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import pickle
import numpy as np
from torch_sparse import SparseTensor, matmul # Import matmul directly

class LearnableWeightedRGCN2(nn.Module):
    def __init__(self, num_nodes, embedding_dim, graph_data_list, node_emb=None, active_indices=None, num_hops=2, num_bases = 4):
        """
        Args:
            num_hops (int): Number of propagation steps per relation (Default: 2).
                            Allows the model to see 'neighbors of neighbors'.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.pert_emb_dim = embedding_dim if node_emb is None else node_emb.weight.shape[-1]
        self.num_hops = num_hops  # <--- NEW: Hop parameter

        # --- A. ACTIVE INDICES OPTIMIZATION ---
        if active_indices is None or node_emb is None:
            active_indices = torch.arange(num_nodes, dtype=torch.long)
            
        self.register_buffer('active_indices', active_indices)
        self.num_active_nodes = len(active_indices)

        # 1. PRE-TRANSFORM
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.pert_emb_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # 2. PREPARE RELATIONS
        self.num_relations = len(graph_data_list)
        for i, (edge_index, edge_weight) in enumerate(graph_data_list):
            if edge_index.numel() == 0:
                self.register_buffer(f'rowptr_{i}', torch.empty(0, dtype=torch.long))
                self.register_buffer(f'col_{i}', torch.empty(0, dtype=torch.long))
                self.register_buffer(f'val_{i}', torch.empty(0, dtype=torch.float))
                continue

            # Note: We keep add_self_loops=False here because we have a dedicated self-loop weight later
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                 sparse_sizes=(self.num_active_nodes, self.num_active_nodes))
            rowptr, col, val = adj_t.csr()
            self.register_buffer(f'rowptr_{i}', rowptr)
            self.register_buffer(f'col_{i}', col)
            self.register_buffer(f'val_{i}', val)

        # 3. RELATION WEIGHTS
        self.num_bases = num_bases
        
        if num_bases is not None and num_bases < self.num_relations:
            # OPTION A: Parameter Efficient Basis Decomposition
            self.use_basis = True
            # The shared bases: [Num_Bases, Dim, Dim]
            self.weight_bases = nn.Parameter(torch.empty(num_bases, embedding_dim, embedding_dim))
            # The coefficients: [Num_Relations, Num_Bases]
            self.comp_coeffs = nn.Parameter(torch.empty(self.num_relations, num_bases))
            
            # Initialize
            nn.init.xavier_uniform_(self.weight_bases)
            nn.init.xavier_uniform_(self.comp_coeffs)
            print(f"RGCN: Using Basis Decomposition ({num_bases} bases for {self.num_relations} relations).")
        else:
            # OPTION B: Standard Full Matrices (Original Way)
            self.use_basis = False
            self.relation_weights = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False) 
                for _ in range(self.num_relations)
            ])
        
        # 4. SEMANTIC ATTENTION COMPONENTS
        self.semantic_attn_vector = nn.Parameter(torch.empty(1, embedding_dim))
        nn.init.xavier_uniform_(self.semantic_attn_vector)
        self.semantic_act = nn.Tanh() # <--- We will actually use this now

        # 5. REST OF MODEL
        self.self_loop_weight = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, self.pert_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
        else:
            self.node_emb = node_emb
        
        self.final_norm = nn.LayerNorm(embedding_dim)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, 
                              missing_keys, unexpected_keys, error_msgs):
        """Hugging Face Hook for resizing buffers on load."""
        for name, buffer in self._buffers.items():
            key = prefix + name
            if key in state_dict:
                input_buffer = state_dict[key]
                if buffer.shape != input_buffer.shape:
                    self._buffers[name] = torch.empty_like(input_buffer)
        
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, 
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, target_node_indices=None):
        x_all = self.node_emb.weight[self.active_indices]
        x = self.pre_encoder(x_all)
        #self_loop = self.self_loop_weight(x)
        messages_list = []
        
        # --- 1. COLLECT MESSAGES (Multi-Hop) ---
        if self.use_basis:
            # Reconstruct the full weight matrices from bases
            # Equation: W_r = sum(a_rb * V_b)
            # Shapes: [R, B] x [B, D, D] -> [R, D, D]
            w_all = torch.einsum('rb, bio -> rio', self.comp_coeffs, self.weight_bases)

        for i in range(self.num_relations):
            rowptr = getattr(self, f'rowptr_{i}')
            col = getattr(self, f'col_{i}')
            val = getattr(self, f'val_{i}')
            
            # Reconstruct SparseTensor
            adj_t = SparseTensor(rowptr=rowptr, col=col, value=val, 
                                 sparse_sizes=(self.num_active_nodes, self.num_active_nodes),
                                 is_sorted=True)
            
            # A. Transform Features (W_r * h)
            # --- APPLY RELATION WEIGHT ---
            if self.use_basis:
                # Use the reconstructed weight matrix for this relation
                w_rel = w_all[i] # [Dim, Dim]
                # Linear transform: x @ W.T
                x_rel = x @ w_rel.transpose(0, 1)
            else:
                # Use the standard linear layer
                x_rel = self.relation_weights[i](x)
            
            # B. Propagate K Hops (A_r * ... * A_r * x_rel)
            # This expands the receptive field for this specific relation
            for _ in range(self.num_hops):  # <--- NEW: Propagation Loop
                x_rel = matmul(adj_t, x_rel)
                
            messages_list.append(x_rel)
            
        # Stack messages: [Num_Nodes, Num_Relations, Emb_Dim]
        stacked_messages = torch.stack(messages_list, dim=1)
        
        # --- 2. CALCULATE ATTENTION ON MESSAGES (Non-Linear) ---
        # Apply Tanh BEFORE the dot product. 
        # Previously, this was linear (stacked_messages * vector), which limited expressivity.
        attention_input = self.semantic_act(stacked_messages) # <--- NEW: Apply Tanh
        
        # Calculate scores
        attention_scores = (attention_input * self.semantic_attn_vector).sum(dim=2) # [N, R]
        
        # Masking logic (unchanged)
        mask = (stacked_messages != 0).any(dim=2)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attention_scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # --- 3. WEIGHTED AGGREGATION ---
        alpha = attn_weights.unsqueeze(-1)
        out_accum = (stacked_messages * alpha).sum(dim=1)

        # --- 4. FINAL PROCESSING ---
        
        out = out_accum + self.bias
        out = F.gelu(out)
        
        # --- SCATTER BACK TO GLOBAL ---
        final_global = torch.zeros(self.num_nodes, self.embedding_dim, device=out.device, dtype=out.dtype)
        final_global[self.active_indices] = out
        
        attn_weights_global = torch.zeros(self.num_nodes, self.num_relations, device=attn_weights.device, dtype=attn_weights.dtype)
        attn_weights_global[self.active_indices] = attn_weights
        attn_weights = attn_weights_global

        final = self.final_norm(final_global)
        
        if target_node_indices is not None:
            return {'final_embs':final[target_node_indices], 'attn_weights':attn_weights,'init_embs': x_all}
        return {'final_embs':final, 'attn_weights':attn_weights,'init_embs': x_all}