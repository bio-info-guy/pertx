
from typing import Literal
import os
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F

def load_pert_embedding_from_gears(gears_path, adata, 
                                  intersect_type : Literal["common","gears"] = "common"):
    """
    load pretrained perturbation embeddings from GEARS model
    Args:
        gears_path: path to gears model
        adata:    scanpy object
        intersect_type:    choose the way to handle gears perturbed genes and adata.obs['genotype']. default intersect.
    Returns:
    """
    # load two required files
    gears_model_dict = torch.load(os.path.join(gears_path, 'model.pt'), map_location = torch.device('cpu'))
    pklfile=os.path.join(gears_path, 'pert_gene_list.pkl')
    import pickle
    with open(pklfile, 'rb') as f:
        pkl_data = pickle.load(f)
    #len(pkl_data['pert_gene_list'])
    gears_gene_list=pkl_data['pert_gene_list']

    # extract the perturbation column in adata
    a_genotype_list=adata.obs['genotype'].unique()
    # a_genotype_list
    import numpy as np

    if intersect_type == 'common':
        #common_genotype_list = a_genotype_list[a_genotype_list.isin(gears_gene_list)]
        common_genotype_list = np.intersect1d(a_genotype_list, gears_gene_list)
    elif intersect_type == 'gears':
        common_genotype_list = gears_gene_list
    else:
        common_genotype_list = gears_gene_list

    print('common genes between GEARS embeddings and your adata genotypes: ' + ','.join(common_genotype_list))
    #print('adata genotypes not in GEARS embeddings: ' + ','.join(a_genotype_list[~a_genotype_list.isin(gears_gene_list)]))
    print('adata genotypes not in GEARS embeddings: ' + ','.join([x for x in a_genotype_list if x not in gears_gene_list]))
    #print('adata genotypes not in GEARS embeddings: ' + ','.join(list(set(a_genotype_list) - set(gears_gene_list))))


    import numpy as np
    common_genotype_indices = [gears_gene_list.index(genotype) for genotype in common_genotype_list if genotype in gears_gene_list]
    gears_model_subset = gears_model_dict['pert_emb.weight'][common_genotype_indices,:]
    genotype_index_gears = {genotype: index for index, genotype in enumerate(common_genotype_list)}
    #genotype_index_gears

    # add wild-type label
    genotype_index_gears['WT'] = len(genotype_index_gears)

    # Add a row of zeros at the end of weights so it represents "WT"
    gears_model_subset = np.vstack([gears_model_subset, np.zeros(gears_model_subset.shape[1])])
    # subset adata
    adata_subset = adata[adata.obs['genotype'].isin(genotype_index_gears.keys())]
    # construct return dict
    ret_dict = {'pert_embedding':gears_model_subset,
                'adata_subset':adata_subset,
                'genotype_index_gears':genotype_index_gears,}
    return ret_dict



def load_pert_embedding_to_model(o_model, model_weights, requires_grad = True):
    """
    load pretrained perturbation embeddings to model
    Args:
      omodel: model object
      model_weights: perturbation embeddings, must be a tensor or numpy array whose size is the same as o_model.pert_encoder.embedding.weight.shape
    Returns:
      model: model object with perturbation embeddings loaded
    """
    model_weights_tensor = torch.tensor(model_weights, 
                                             dtype=o_model.pert_encoder.embedding.weight.dtype, 
                                             device=o_model.pert_encoder.embedding.weight.device)
    if model_weights_tensor.shape != o_model.pert_encoder.embedding.weight.shape:
        raise ValueError(f"model_weights_tensor.shape {model_weights_tensor.shape} does not equal to o_model.pert_encoder.embedding.weight.shape {o_model.pert_encoder.embedding.weight.shape}")
    o_model.pert_encoder.embedding.weight.data = model_weights_tensor # torch.tensor(gears_model_subset,dtype=torch.double)
    o_model.pert_encoder.embedding.weight.requires_grad = requires_grad
    return o_model


def extract_master_edge_index(rgcn_module, to_cpu=True):
    """
    Reconstructs master_edge_index and FILTERS OUT self-loops.
    Safe to use regardless of whether add_self_loops was True or False.
    """
    all_rows = []
    all_cols = []
    
    print("Extracting edges and filtering self-loops...")
    
    for i in range(rgcn_module.num_relations):
        # 1. Retrieve buffers
        rowptr = getattr(rgcn_module, f'rowptr_{i}')
        col = getattr(rgcn_module, f'col_{i}')
        
        # 2. Reconstruct SparseTensor
        adj_t = SparseTensor(
            rowptr=rowptr, 
            col=col, 
            sparse_sizes=(rgcn_module.num_active_nodes, rgcn_module.num_active_nodes),
            is_sorted=True
        )
        
        # 3. Get Coordinate format (Row, Col)
        row, col_indices, _ = adj_t.coo()
        
        if to_cpu:
            row = row.cpu()
            col_indices = col_indices.cpu()
            
        # --- THE FIX: FILTER SELF-LOOPS ---
        # We create a boolean mask: Keep only where row IS NOT equal to col
        mask = row != col_indices
        
        # Apply mask to keep only real connections
        row = row[mask]
        col_indices = col_indices[mask]
        # ----------------------------------
            
        all_rows.append(row)
        all_cols.append(col_indices)

    # 4. Concatenate
    master_row = torch.cat(all_rows)
    master_col = torch.cat(all_cols)
    
    master_edge_index = torch.stack([master_row, master_col], dim=0)
    
    # Optional: Remove duplicates (if Graph 1 and Graph 2 share the same edge)
    # For link prediction, duplicates are usually fine (weights the edge higher),
    # but if you want a strict binary "exists or not", uncomment below:
    # master_edge_index = torch.unique(master_edge_index, dim=1)
    
    print(f"Done. Extracted {master_edge_index.size(1)} edges (excluding self-loops).")
    return master_edge_index


import pandas as pd
import torch
from torch_geometric.utils import coalesce

def process_string_dataframe_optimized(df, local_node_map, graph_cols, min_weight=100):
    """
    Converts STRING dataframe to PyG edge indices using a pre-computed local map.
    """
    # 1. Map String Names to Local Integers (0 to N_graph-1)
    # Using .map is fast. Drop edges where genes aren't in the map (if any).
    df['u'] = df['gene1'].map(local_node_map)
    df['v'] = df['gene2'].map(local_node_map)
    
    # Filter out edges where one node is missing from the map
    valid_edges = df.dropna(subset=['u', 'v'])
    
    if len(valid_edges) < len(df):
        print(f"Dropped {len(df) - len(valid_edges)} edges involving genes not in the graph universe.")
    
    u = torch.tensor(valid_edges['u'].values, dtype=torch.long)
    v = torch.tensor(valid_edges['v'].values, dtype=torch.long)
    num_nodes = len(local_node_map)

    graph_data_list = []
    
    # 2. Iterate Cols and Build Tensors
    for col in graph_cols:
        if col not in df.columns:
            continue
            
        weights = torch.tensor(valid_edges[col].values, dtype=torch.float)
        
        # Filter by weight threshold
        mask = weights > min_weight
        if not mask.any():
            # Handle empty graph case
            graph_data_list.append((torch.empty((2, 0), dtype=torch.long), torch.empty(0)))
            continue

        curr_u = u[mask]
        curr_v = v[mask]
        curr_w = weights[mask] / 1000.0 # Normalize 0-1000 -> 0-1
        
        edge_index = torch.stack([curr_u, curr_v], dim=0)
        
        # Symmetrize and Coalesce (Undirected Graph)
        edge_index, edge_weight = coalesce(edge_index, curr_w, num_nodes=num_nodes)
        
        graph_data_list.append((edge_index, edge_weight))
        
    return graph_data_list

def get_graph_and_vocab_unified(
    gene_list: list, 
    string_file: str, 
    info_file: str,
    shared_vocab: bool = True,
    graph_cols: list = None
):
    """
    Orchestrates the creation of Vocab, Active Indices, and Graph Tensors.
    
    Args:
        gene_list: List of genes from your expression data (AnnData).
        string_file: Path to STRING links.
        info_file: Path to STRING protein info.
        shared_vocab: 
            If True (Scenario B): Vocab is Union(Expression, Graph). Returns active_indices.
            If False (Scenario A): Vocab is Expression only. Graph has its own vocab. Returns active_indices=None.
    """
    if graph_cols is None:
        graph_cols = ['neighborhood_transferred', 'fusion', 'cooccurence', 
        'homology', 'coexpression', 'coexpression_transferred', 
        'experiments', 'experiments_transferred', 'database', 
        'database_transferred', 'textmining', 'textmining_transferred']

    # --- STEP 1: LOAD RAW ENTITIES (No processing yet) ---
    print("Loading STRING DB raw data...")
    b = pd.read_csv(info_file, sep='\t', usecols=['#string_protein_id', 'preferred_name'])
    prot_to_gene = pd.Series(b.preferred_name.values, index=b['#string_protein_id']).to_dict()
    
    a = pd.read_csv(string_file, sep=' ')
    a['gene1'] = a['protein1'].map(prot_to_gene)
    a['gene2'] = a['protein2'].map(prot_to_gene)
    
    # Get the universe of genes existing in the Graph
    graph_genes_unique = sorted(list(set(a['gene1']) | set(a['gene2'])))
    
    # --- STEP 2: BUILD MASTER VOCAB ---
    # We use a set to ensure uniqueness
    expression_genes_set = set(gene_list)
    
    if shared_vocab:
        # SCENARIO B: Union of Expression and Graph
        # The Master Vocab must contain everything
        combined_genes = sorted(list(expression_genes_set | set(graph_genes_unique)))
        
        # Build Master Vocab
        # Note: Add special tokens logic here if needed (e.g. MasterVocab class)
        master_vocab_list = ['<pad>', '<cls>', '<unk>', 'WT'] + combined_genes
        master_vocab = {token: i for i, token in enumerate(master_vocab_list)}
        
        # Graph Universe is exactly the graph genes we found
        final_graph_genes = graph_genes_unique
        
    else:
        # SCENARIO A: Separate Vocabs
        # Master Vocab is ONLY based on expression data
        master_vocab_list = ['<pad>', '<cls>', '<unk>', 'WT'] + sorted(list(expression_genes_set))
        master_vocab = {token: i for i, token in enumerate(master_vocab_list)}
        
        # Graph Universe is just the graph genes (independent)
        final_graph_genes = graph_genes_unique

    # --- STEP 3: GENERATE MAPPINGS & INDICES ---
    
    # Map 1: Local Map (Gene Name -> 0..N_graph)
    # Used to build the adjacency matrix efficiently
    local_node_map = {name: i for i, name in enumerate(final_graph_genes)}
    
    # Map 2: Active Indices (0..N_graph -> Global_Vocab_ID)
    if shared_vocab:
        # We need to map every gene in the local graph to its ID in the Master Vocab
        # This tells the RGCN: "Row 0 of Adjacency corresponds to Embedding Index 5042"
        indices = []
        for name in final_graph_genes:
            indices.append(master_vocab[name])
        active_indices = torch.tensor(indices, dtype=torch.long)
    else:
        # In separate mode, the GNN has its own embedding matrix of size N_graph.
        # It doesn't need to look up into the Master Vocab.
        active_indices = None 

    # --- STEP 4: PROCESS GRAPH ---
    print(f"Building Adjacency Matrices for {len(final_graph_genes)} nodes...")
    adj_graphs = process_string_dataframe_optimized(
        a, 
        local_node_map, 
        graph_cols
    )

    return master_vocab, adj_graphs, active_indices, local_node_map