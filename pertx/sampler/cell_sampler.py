from abc import ABC, abstractmethod
import numpy as np

class PairingStrategy(ABC):
    @abstractmethod
    def get_target(self, source_indices, source_metadata):
        """
        Args:
            source_indices: Array of global indices (the 'current' cells)
            source_metadata: DataFrame/Dict of info (cell_type, genotype) for these cells
        Returns:
            target_indices: Array of global indices (the 'next' cells)
            target_metadata: Metadata for the targets
            weights: (Optional) If we are doing weighted mixing (barycenters)
        """
        pass
    
    @abstractmethod
    def prepare(self, data_source):
        """Run any necessary setup (building dicts, loading maps)"""
        pass

class RandomGenotypeStrategy(PairingStrategy):
    def __init__(self, meta, perturb_col='genotype', ctrl_label = 'WT', context_col = 'celltype'):
        self.perturb_col= perturb_col
        self.ctrl_label = ctrl_label
        self.context_col = context_col
        self.df = meta
        self.df['idx'] = self.df.indices
        self.groups = self.df.groupby([self.context_col, self.perturb_col])['idx'].apply(np.array).to_dict()

    def get_target(self, source_indices, source_metadata):
        targets = []
        for i, idx in enumerate(source_indices):
            ctype = source_metadata.iloc[i][self.context_col]
            curr_geno = source_metadata.iloc[i][self.perturb_col]
            
            # Logic: If WT, go to Random Pert. If Pert, stay Pert? 
            # (Adapting your original logic here)
            if curr_geno == self.ctrl_label:
                # Pick a random perturbation valid for this cell type
                # (You can refine this selection logic)
                possible_keys = [k for k in self.groups.keys() if k[0] == ctype and k[1] != self.ctrl_label]
                if not possible_keys:
                    targets.append(idx) # Fallback to self
                    continue
                target_key = possible_keys[np.random.randint(len(possible_keys))]
            else:
                # If already perturbed, maybe we map to another cell of SAME perturbation?
                target_key = (ctype, curr_geno)

            # Sample the specific cell index
            candidates = self.groups.get(target_key, [idx])
            targets.append(np.random.choice(candidates))
            
        return np.array(targets), None
    
class PrecomputedOTStrategy(PairingStrategy):
    def __init__(self, mapping_file_path):
        self.mapping_file_path = mapping_file_path
        self.mapping_table = None # Will hold the lookup table

    def prepare(self, data_source):
        # Load the pre-computed map.
        # Format: A simple array where index=source_cell, value=target_cell
        # Or a sparse matrix if doing weighted averages.
        print(f"Loading Offline OT Map from {self.mapping_file_path}...")
        self.mapping_table = np.load(self.mapping_file_path) 
        # e.g., mapping_table[100] = 502 (Cell 100 maps to Cell 502)

    def get_target(self, source_indices, source_metadata):
        # Extremely fast O(1) lookup
        target_indices = self.mapping_table[source_indices]
        return target_indices, None