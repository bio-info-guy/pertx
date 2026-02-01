from typing import List, Tuple, Dict, Union, Optional, Literal
import os
import pickle
from operator import itemgetter
from sklearn.model_selection import train_test_split, KFold
import torch
import numpy as np
import random
from scipy.sparse import issparse
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from anndata import AnnData
import anndata
from scipy.sparse import issparse
from sklearn.model_selection._split import _BaseKFold
from ..tokenizer.custom_tokenizer import tokenize_and_pad_batch, random_mask_value, MasterVocab
from ..sampler.ot import compute_ot_for_subset


def add_batch_info(adata):
    """helper function to add batch effect columns into adata"""
    if "batch" not in adata.obs.columns: 
        batch_ids_0=random.choices( [0,1], k=adata.shape[0])
        adata.obs["batch"]=batch_ids_0
    if "batch_id" not in adata.obs.columns: 
        adata.obs["str_batch"] = adata.obs["batch"]
        adata.obs["str_batch"] = adata.obs["str_batch"].astype(str)
        adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values

"""
STEPS TO TRAIN:
create a PertTFDataManager First and use it generate loaders, validation and data_gen dictionary
Pass train_loader, valid_loader and data_gen to the wrapper_train either once or as part of kfold loop
"""

class PertTFDataset(Dataset):
    """
    A PyTorch Dataset for AnnData objects with Integrated Optimal Transport Sampling.
    """
    def __init__(self, 
                 adata, 
                 indices: np.ndarray = None, 
                 # OT Parameters
                 use_ot: bool = False,
                 ot_pickle_path: str = None,
                 ot_top_k: int = 10,
                 ot_epsilon: Union[float, str] = "auto",
                 ot_max_dist: Union[float, str] = "auto",
                 # Standard Parameters
                 cell_type_to_index: dict = None, 
                 genotype_to_index: dict = None, 
                 expr_layer: str = 'X_binned',
                 ps_columns: list = None, 
                 ps_columns_perturbed_genes: list = None, 
                 next_cell_pred: str = "identity", 
                 additional_ps_dict: dict = None, 
                 only_sample_wt_pert: bool = False,
                 sample_once = False):
        
        self.adata = adata
        self.sample_once = sample_once
        self.cell2cell_dict = {}
        self._check_anndata_content()
        self.indices = indices if indices is not None else np.arange(len(self.adata.obs.index))
        
        # OT Configuration
        self.use_ot = use_ot
        self.ot_pickle_path = ot_pickle_path
        self.ot_top_k = ot_top_k
        self.ot_epsilon = ot_epsilon
        self.ot_max_dist = ot_max_dist
        self.ot_cache = {} # Holds the map for the CURRENT indices
        
        self.expr_layer = expr_layer
        self.next_cell_pred = next_cell_pred

        # Mappings
        self.cell_type_to_index = cell_type_to_index if cell_type_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())}
        self.genotype_to_index = genotype_to_index if genotype_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())}
        
        # PS Scores Setup
        self.ps_columns = list(ps_columns) if ps_columns else []
        
        
        # Handle Lochness specific setup

        # Initialize pools and OT
        self.next_cell_dict = self._create_next_cell_pool()
        self.only_sample_wt_pert = only_sample_wt_pert

        if self.next_cell_pred == "lochness":
            self._setup_lochness(ps_columns, ps_columns_perturbed_genes, additional_ps_dict)
        self.ps_matrix = self.adata.obs[self.ps_columns].values.astype(np.float32) if self.ps_columns else np.zeros((self.adata.shape[0],1), dtype=np.float32)
        # Trigger initial OT calculation if enabled
        if self.use_ot and self.next_cell_pred == "pert":
            self._recalculate_ot()

    def _setup_lochness(self, ps_columns, ps_columns_perturbed_genes, additional_ps_dict):
        """Helper to organize Lochness config."""
        if ps_columns is None:
             raise ValueError("PS columns must be provided for lochness prediction")
        if len(ps_columns) != len(ps_columns_perturbed_genes):
            raise ValueError("ps_columns_perturbed_genes length mismatch")
        
        ps_columns_perturbed_genes = [x for x in ps_columns_perturbed_genes if x in self.genotype_to_index]
        if len(ps_columns_perturbed_genes) != len(ps_columns):
            print('Specified perturbed genes for PS column after filtering:' + ','.join(ps_columns_perturbed_genes))
            raise ValueError("The ps_columns_perturbed_genes must be specified and has to have equal length as ps_matrix")
        self.ps_columns_perturbed_genes = ps_columns_perturbed_genes

        if additional_ps_dict is None:
            self.additional_ps_dict={}
            self.additional_ps_names=[]
        else:
            self.additional_ps_dict = additional_ps_dict
            self.additional_ps_names = list(additional_ps_dict.keys())
            for x in self.additional_ps_names:
                if x not in self.genotype_to_index:
                    raise ValueError(f"Gene {x} in additional_ps_dict not in genotype index.")

    def _check_anndata_content(self):
        assert 'genotype' in self.adata.obs.columns and 'celltype' in self.adata.obs.columns, 'no genotype or celltype column found in anndata'
        add_batch_info(self.adata)

    def set_new_indices(self, indices, next_cell_pool = True):
        """
        Public method to update the dataset split (e.g. for K-Fold).
        Automatically recalculates OT maps for the new subset.
        """
        self.indices = indices
        # Clear per-epoch static cache if it exists
        self.cell2cell_dict = {} 
        
        if next_cell_pool:
            self.next_cell_dict = self._create_next_cell_pool()

        if self.use_ot and self.next_cell_pred == "pert":
            print(f"--- Indices Updated (N={len(indices)}). Recalculating OT Maps... ---")
            self._recalculate_ot()

    def get_adata_subset(self, next_cell_pred = 'identity'):
        assert next_cell_pred in ['pert', 'identity', "lochness"], 'next_cell_pred can only be identity or pert or lochness'

        adata_small = self.adata[self.indices,].copy()
        if next_cell_pred == "identity" :
            return adata_small
        elif next_cell_pred == "lochness":
            adata_small.obs['genotype_next'] = adata_small.obs['genotype']
            return adata_small
        else:
            #adata_small = self.adata[self.indices,].copy()
            next_cell_id_list = []
            next_pert_list = []
            next_cell_global_idx_list = []
            for i in self.indices:
                current_cell_idx = self.adata.obs.index[i]
                current_cell_celltype = self.adata.obs.at[current_cell_idx, 'celltype']
                current_cell_genotype = self.adata.obs.at[current_cell_idx, 'genotype']
                next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_idx, current_cell_celltype, current_cell_genotype)
                next_cell_id_list.append(next_cell_id)
                next_pert_list.append(next_pert_label_str)
                next_cell_global_idx_list.append(self.adata.obs.index.get_loc(next_cell_id))
            adata_small.obs['genotype_next'] = next_pert_list
            adata_small.obs['next_cell_id'] = next_cell_id_list
            adata_small.layers['next_expr'] = self.adata.layers[self.expr_layer][next_cell_global_idx_list]
        return adata_small
    

    def _recalculate_ot(self):
        # 1. Create View
        adata_subset = self.adata[self.indices]
        
        # --- NEW: Auto-Calibration Logic ---
        current_threshold = self.ot_max_dist
        
        # -----------------------------------

        # 2. Compute Maps (Pass the calculated threshold)
        new_maps = compute_ot_for_subset(
            adata_subset, 
            top_k=self.ot_top_k, 
            epsilon=self.ot_epsilon,
            max_dist_sq=current_threshold, # <--- Use the variable, not self.ot_max_dist
            pca_key='X_pca'
        )
        
        # 3. Update Cache & Pickle (Same as before)
        self.ot_cache = new_maps
        
        if self.ot_pickle_path:
            full_cache = {}
            if os.path.exists(self.ot_pickle_path):
                try:
                    with open(self.ot_pickle_path, 'rb') as f:
                        full_cache = pickle.load(f)
                except Exception:
                    pass # handle read error
            
            full_cache.update(new_maps)
            try:
                with open(self.ot_pickle_path, 'wb') as f:
                    pickle.dump(full_cache, f)
            except Exception as e:
                print(f"Warning: Could not save OT pickle: {e}")


    def __len__(self):
        return len(self.indices)

    def _create_next_cell_pool(self):
        """Pre-computes a dictionary for fast sampling of random cells."""
        if self.next_cell_pred != "pert":
            return None
            
        next_cell_dict = {}
        # View of the current subset
        obs_subset = self.adata.obs.iloc[self.indices]
        
        for cell_type in obs_subset['celltype'].unique():
            next_cell_dict[cell_type] = {}
            for genotype in obs_subset['genotype'].unique():
                # Get names of cells matching criteria
                mask = (obs_subset['celltype'] == cell_type) & (obs_subset['genotype'] == genotype)
                included_cells_indices = obs_subset[mask].index.tolist()
                if included_cells_indices:
                    next_cell_dict[cell_type][genotype] = included_cells_indices
        return next_cell_dict

    def _sample_next_cell(self, current_cell_idx, current_cell_celltype, current_cell_genotype):
        """
        Selects the next cell pair. 
        current_cell_idx: String (Cell Name)
        """
        if self.next_cell_pred == "identity" or self.next_cell_pred == "lochness":
            return current_cell_idx, current_cell_genotype

        # 1. Determine Target Genotype
        valid_genotypes = self.next_cell_dict.get(current_cell_celltype, {})
        if not valid_genotypes: return current_cell_idx, current_cell_genotype 

        if current_cell_genotype == 'WT':
            # WT cells try to map to a random perturbation
            next_pert_value = random.choice(list(valid_genotypes.keys()))
        else:
            # Perturbed cells map to themselves (logic handled later)
            next_pert_value = current_cell_genotype

        # 2. OT Sampling (WT -> Perturbation only)
        if self.use_ot and current_cell_genotype == 'WT' and next_pert_value != 'WT':
            # Check if we have a valid OT map for this specific cell and target
            if current_cell_idx in self.ot_cache:
                if next_pert_value in self.ot_cache[current_cell_idx]:
                    candidates, weights = self.ot_cache[current_cell_idx][next_pert_value]
                    # Sample weighted choice
                    next_cell_id = random.choices(candidates, weights=weights, k=1)[0]
                    return next_cell_id, next_pert_value
                else:
                    return current_cell_idx, current_cell_genotype
            else:
                return current_cell_idx, current_cell_genotype
        # 3. Fallback / Standard Logic
        if next_pert_value == current_cell_genotype and self.only_sample_wt_pert:
            # Perturbed cells (or WT mapped to WT) return themselves
            return current_cell_idx, next_pert_value
        else:
            # Random sampling from the pool
            possible_next_cells = valid_genotypes.get(next_pert_value, [current_cell_idx])
            next_cell_id = random.choice(possible_next_cells)
            return next_cell_id, next_pert_value
    def __getitem__(self, idx: int):
        # 1. Get Global Index and Metadata
        current_cell_global_idx = self.indices[idx]
        current_cell_idx = self.adata.obs.index[current_cell_global_idx] # This is the Name (String)
        
        # Fast access using .at for scalar lookups
        current_cell_celltype = self.adata.obs.at[current_cell_idx, 'celltype']
        current_cell_genotype = self.adata.obs.at[current_cell_idx, 'genotype']
        current_cell_batch_label = self.adata.obs.at[current_cell_idx, 'batch_id']

        # 2. Sample Next Cell
        if self.sample_once:
            if current_cell_idx in self.cell2cell_dict:
                next_cell_id, next_pert_label_str = self.cell2cell_dict[current_cell_idx]
            else:
                next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_idx, current_cell_celltype, current_cell_genotype)
                self.cell2cell_dict[current_cell_idx] = (next_cell_id, next_pert_label_str)
        else:
            next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_idx, current_cell_celltype, current_cell_genotype)
        
        # 3. Resolve Next Cell to Global Index
        # We need the integer location to fetch expression data
        next_cell_global_idx = self.adata.obs.index.get_loc(next_cell_id)

        # 4. Fetch Expression Data
        # Flatten sparse matrices if necessary
        current_expr = self.adata.layers[self.expr_layer][current_cell_global_idx]
        if issparse(current_expr): current_expr = current_expr.toarray().flatten()

        next_expr = self.adata.layers[self.expr_layer][next_cell_global_idx]
        if issparse(next_expr): next_expr = next_expr.toarray().flatten()

        # 5. Prepare Labels
        cell_label = self.cell_type_to_index[current_cell_celltype]
        pert_label = self.genotype_to_index[current_cell_genotype]
        pert_label_next = self.genotype_to_index[next_pert_label_str]
        
        # 6. Prepare PS Scores
        ps_scores = self.ps_matrix[current_cell_global_idx]
        ps_scores_next = self.ps_matrix[next_cell_global_idx]

        # 7. Lochness Logic (Overrides next ps/pert labels)
        if self.next_cell_pred == "lochness":
            selection_pool_length = len(self.ps_columns_perturbed_genes) + len(self.additional_ps_names)
            random_pert_ind = random.randint(0, selection_pool_length-1)
            
            if random_pert_ind < len(self.ps_columns_perturbed_genes): 
                pert_label_next = self.genotype_to_index[self.ps_columns_perturbed_genes[random_pert_ind]] 
                ps_scores_next = np.array([ps_scores[random_pert_ind]], dtype=np.float32) 
            else:
                selected_gene = self.additional_ps_names[random_pert_ind - len(self.ps_columns_perturbed_genes)] 
                pert_label_next = self.genotype_to_index[selected_gene]
                ps_scores_next = np.array([self.additional_ps_dict[selected_gene]], dtype=np.float32) 

        return {
            "expr": current_expr,
            "expr_next": next_expr,
            "genes": self.adata.var.index,
            "next_genes": self.adata.var.index,
            "celltype_labels": cell_label,
            "perturbation_labels": pert_label,
            "batch_labels": current_cell_batch_label,
            "celltype_labels_next": cell_label, # Cell type invariant
            "perturbation_labels_next": pert_label_next,
            "ps": ps_scores,
            "ps_next": ps_scores_next,
            "index": current_cell_global_idx,
            "next_index": next_cell_global_idx,
            'name': current_cell_idx,
            'next_name': next_cell_id
        }

class PertBatchCollator:
    """
    A collate function for the DataLoader that tokenizes, pads, and masks batches on the fly.
    """
    def __init__(self, vocab: object, gene_ids: np.ndarray, full_tokenize: bool = False, hvg_inds = None, **config):
        self.config = config
        self.vocab = vocab
        self.gene_ids = gene_ids # vector of gene ids
        self.full_tokenize = full_tokenize
        self.include_zero_gene = config.get('include_zero_gene', True)
        self.append_cls = config.get('append_cls', True)
        self.cls_value = config.get('cls_value', -3)
        self.cls_token = config.get('cls_token', '<cls>')
        self.max_seq_len = config.get('max_seq_len', 3000)
        self.pad_token = config.get('pad_token', '<pad>')
        self.pad_value = config.get('pad_value', -2)
        self.mask_ratio = config.get('mask_ratio', 0.15)
        self.mask_value = config.get('mask_value', -1)
        self.nonzero_prop = config.get('nonzero_prop', 0.7)
        self.sampling_mode = config.get('sampling_mode', 'simple')
        self.fix_nonzero_prop = config.get('fix_nonzero_prop', False)
        self.non_hvg_size = min(config.get('non_hvg_size', 1000), len(hvg_inds[1])) if hvg_inds is not None else 0
        self.hvg_inds = hvg_inds

    def __call__(self, batch: list) -> dict:
        """
        Processes a list of samples from the Dataset into a single batch tensor.
        """

        # Define which items to get from each dictionary
        get_values = itemgetter('expr', 'expr_next', 'genes', 'next_genes')
        # Create an iterator of tuples, then unzip them into four separate lists
        expr_list, expr_next_list, gene_list, gene_next_list = map(list, zip(*(get_values(item) for item in batch)))

        # 2. Tokenize and pad the expression data for the current batch
        # max seq len determines the context window for pertTF transformer modeling
        # during validation and predictions, this window may be around all genes with expression
        max_seq_len = self.max_seq_len if not self.full_tokenize else len(self.gene_ids) + self.append_cls

        # TODO: These functions may need to be modified to accomodate inputs w differing number of genes in the future
        expr_mat, expr_mat_next = np.array(expr_list), np.array(expr_next_list)
        tokenized, gene_idx_list = tokenize_and_pad_batch(
            expr_mat, self.gene_ids, max_len=max_seq_len,cls_token=self.cls_token,
            vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
            append_cls=self.append_cls, include_zero_gene=self.include_zero_gene, 
            cls_value=self.cls_value, sampling_mode = self.sampling_mode,
            fix_nonzero_prop=self.fix_nonzero_prop, nonzero_prop=self.nonzero_prop,
            hvg_inds = self.hvg_inds, non_hvg_size= self.non_hvg_size
        )
        tokenized_next, _ = tokenize_and_pad_batch(
            expr_mat_next, self.gene_ids, max_len=max_seq_len, cls_token=self.cls_token,
            vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
            append_cls=self.append_cls, include_zero_gene=self.include_zero_gene, 
            sample_indices=gene_idx_list, 
            cls_value=self.cls_value, sampling_mode = self.sampling_mode,
            fix_nonzero_prop=self.fix_nonzero_prop, nonzero_prop=self.nonzero_prop,
            hvg_inds = self.hvg_inds, non_hvg_size= self.non_hvg_size
        )
        
        # 3. Apply random masking for this batch
        masked_values = random_mask_value(
            tokenized["values"], mask_ratio=self.mask_ratio,
            mask_value=self.mask_value, pad_value=self.pad_value,
            cls_value= self.cls_value
        )

        # 4. Collate all other labels into tensors
        full_gene_id = torch.from_numpy(self.gene_ids).long()
        collated_batch = {
            "gene_ids": tokenized["genes"],
            "next_gene_ids": tokenized_next["genes"],
            "values": masked_values,
            "target_values": tokenized["values"],
            "target_values_next": tokenized_next["values"],
            "full_expr": torch.Tensor(expr_mat),
            "full_expr_next": torch.Tensor(expr_mat_next),
            "full_gene_ids": torch.stack([full_gene_id for i in range(len(batch))], dim = 0)
        }
        
        # Stack scalar or vector labels from each item in the batch
        for key in batch[0].keys():
            if 'name' in key:
                values = [item[key] for item in batch]
                collated_batch[key] = values
            elif key not in ["expr", "expr_next"] and key not in ['genes', 'next_genes']:
                values = [item[key] for item in batch]
                tensor = torch.from_numpy(np.array(values))
                # Ensure labels are long type and scores are float
                collated_batch[key] = tensor.long() if 'label' in key else tensor.float()

        return collated_batch
    
    

class PertTFUniDataManager:
    """
    Manages data loading, preprocessing, and splitting using a single (Uni) AnnData object.
    This class encapsulates all data-related setup, including vocab, mappings,
    and provides methods to get data loaders for training and cross-validation.
    """
    def __init__(self, 
                 adata: AnnData, 
                 config: object, 
                 ps_columns: list = None,
                 ps_columns_perturbed_genes: list = None,
                 celltype_to_index: dict = None, 
                 vocab: MasterVocab = None,
                 genotype_to_index: dict = None, 
                 expr_layer: str = 'X_binned',
                 next_cell_pred_type: str = "identity", 
                 additional_ps_dict: dict = None, 
                 only_sample_wt_pert: bool = False):
        #assert not adata.is_view, "The provided anndata is likely a view of the original anndata, this is probably due to slicing the original annadata object, please use the .copy() method to provide a copy"
        self.adata = adata.copy() # make a copy of the data so that no issues arise if adata is a anndata view
        self.indices = np.arange(self.adata.n_obs)
        self.config = config
        self.ps_columns = ps_columns # perhaps this can incorporated into config
        self.ps_columns_perturbed_genes = ps_columns_perturbed_genes
        self.additional_ps_dict = additional_ps_dict
        self.expr_layer = expr_layer
        self.only_sample_wt_pert = config.get('only_sample_wt_pert', only_sample_wt_pert)
        self.next_cell_pred_type = config.get('next_cell_pred_type', next_cell_pred_type)
        # --- Perform one-time data setup --- 
        print("Initializing PertTFUniDataManager: Creating vocab and mappings...")
        #if "batch_id" not in self.adata.obs.columns:
         #   self.adata.obs["str_batch"] = "batch_0"
          #  self.adata.obs["batch_id"] = self.adata.obs["str_batch"].astype("category").cat.codes
        
        # Create and store mappings and vocab as instance attributes
                #self.num_batch_types = len(self.adata.obs["batch_id"].unique())

        self.genes = self.adata.var.index.tolist()
        self.vocab = MasterVocab(self.genes, config.special_tokens) if vocab is None else vocab
        assert self.adata.var.index.isin(self.vocab.stoi).all(), 'Not all genes are in provided vocab, please prefiltered the Anndata first'
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.gene_ids = np.array(self.vocab(self.genes), dtype=int)

        self.set_genotype_index(genotype_to_index= genotype_to_index)
        self.set_celltype_index(celltype_to_index= celltype_to_index)



        self.hvg_inds = None
        n_hvg = config.get('n_hvg', 3000)
        if config.get('sampling_mode', 'simple') == 'hvg':
            self.hvg_col = config.get('hvg_col', 'highly_variable')
            assert self.hvg_col in adata.var.keys(), 'adata must have calculated HVGs or adata.var must have hvg_col'
            n_hvg = min(self.adata.var[self.hvg_col].sum(), n_hvg)
            non_hvg = min(len(self.gene_ids) - n_hvg, config.get('non_hvg_size', 1000))
            self.config.update({'max_seq_len': n_hvg + non_hvg + config.get('append_cls', True)}, allow_val_change=True)
            print(f'sampling_mode is hvg, sampling {n_hvg} HVGs + {non_hvg} non-HVGs for training')
            self.hvg_inds = (np.where(self.adata.var[self.hvg_col])[0], np.where(~self.adata.var[self.hvg_col])[0])
        
        add_batch_info(self.adata)
        self.num_batch_types = len(self.adata.obs["batch_id"].unique())
        # The collators can be created once and reused
        ## first collator is the training collator, with a context window set in config
        self.collator = PertBatchCollator(self.vocab, self.gene_ids, hvg_inds = self.hvg_inds, **config)
        ## full collator may be used for validation or inference 
        ## This may be very slow for full gene set, scaling is roughly 2x context length -> 3.6x time, 3-4x more memory
        self.full_token_collator = PertBatchCollator( self.vocab, self.gene_ids, full_tokenize=True, hvg_inds = self.hvg_inds, **config)
        print("Initialization complete.")

    def set_genotype_index(self, genotype_to_index):
        self.genotype_to_index = {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())} if genotype_to_index is None else genotype_to_index
        self.num_genotypes = len(self.genotype_to_index)

    def set_celltype_index(self, celltype_to_index):
        self.cell_type_to_index = {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())} if celltype_to_index is None else celltype_to_index
        self.num_cell_types = len(self.cell_type_to_index)

    def get_adata_info_dict(self):
        data_gen = { 
            'genes': self.genes,
            'gene_ids': self.gene_ids,
            'vocab': self.vocab,
            'num_batch_types': self.num_batch_types, # need to change this
            'num_cell_types': self.num_cell_types,
            'num_genotypes': self.num_genotypes,
            'cell_type_to_index': self.cell_type_to_index,
            'genotype_to_index': self.genotype_to_index
        }
        if self.ps_columns is not None:
            data_gen['ps_names']=[x for x in self.ps_columns if x in self.adata.obs.columns]
        else:
            data_gen['ps_names']=["PS"]
                
        return data_gen

    def _create_dataset_from_indices(self, indices, sample_once = False):
        """A helper function to create PertTFDataset from underlying adata."""
        perttf_dataset = PertTFDataset(
            self.adata, indices=indices, use_ot=self.config.use_ot, ot_pickle_path=self.config.ot_pickle_path, 
            cell_type_to_index=self.cell_type_to_index, genotype_to_index=self.genotype_to_index,
            ps_columns=self.ps_columns, ps_columns_perturbed_genes = self.ps_columns_perturbed_genes, 
            next_cell_pred=self.next_cell_pred_type ,  additional_ps_dict = self.additional_ps_dict,  
            expr_layer=self.expr_layer, only_sample_wt_pert=self.only_sample_wt_pert, sample_once=sample_once
        )
        return perttf_dataset

    def _create_loaders_from_dataset(self, dataset, full_token_collator = False):
        """A helper function to create dataloaders from PertTFDataset."""    
        collator = self.collator if not full_token_collator else self.full_token_collator 
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=8, collate_fn=collator, pin_memory=True
        )
        return loader

    def get_data_w_loader(self, indices = None, full_data = False, full_token = False, sample_once = False):
        indices = self.indices if indices is None and full_data else indices
        data = self._create_dataset_from_indices(indices, sample_once=sample_once)
        loader = self._create_loaders_from_dataset(data, full_token)
        return data, loader

    def get_train_valid_loaders(self, test_size: float = 0.1, train_indices = None, valid_indices = None, full_token_validate  = False, random_state = None, sample_once = False):
        """Provides a single, standard train/validation split."""
        print(f"Creating a single train/validation split (test_size={test_size})...")
        if train_indices is None or valid_indices is None:
            indices = np.arange(self.adata.n_obs)
            train_indices, valid_indices = train_test_split(indices, test_size=test_size, shuffle=True, random_state=random_state)
        else:
            if len(set(train_indices).intersection(valid_indices)) > 0:
                print('WARNING: training data and validation data are not separate, this may be okay for perturbation if the shared samples are ctrls')
            print('overiding random train/valid split with provided indices')
        train_data, train_loader = self.get_data_w_loader(train_indices, sample_once=sample_once)
        valid_data, valid_loader = self.get_data_w_loader(valid_indices, full_token=full_token_validate, sample_once=sample_once)
        return train_data, train_loader, valid_data, valid_loader, self.get_adata_info_dict()

    def get_k_fold_split_loaders(self, cv = 5):
        """
        An iterator that yields train and validation dataloaders for each fold
        in a k-fold cross-validation setup.
        """
        kf = cv if issubclass(cv.__class__,  _BaseKFold) else KFold(n_splits=cv, shuffle=True)
        print(f"Set up K-Fold cross-validation with {kf.n_splits} folds")
        for fold, (train_indices, valid_indices) in enumerate(kf.split(self.indices)):
            print(f"--- Yielding data loaders for Fold {fold+1}/{kf.n_splits} ---")
            yield self.get_train_valid_loaders(train_indices = train_indices, valid_indices = valid_indices, full_token_validate  = False)

