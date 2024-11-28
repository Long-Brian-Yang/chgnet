from __future__ import annotations
import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from torch.utils.data import DataLoader

# Suppress warnings
warnings.simplefilter("ignore")


def get_project_paths():
    """Get project paths."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths = {
        'structures_dir': os.path.join(root_dir, 'data/structures'),
        'file_path': os.path.join(root_dir, 'data/data_list.csv'),
        'output_dir': os.path.join(root_dir, 'logs'),
    }
    
    # Create directories if they don't exist
    for dir_path in paths.values():
        dir_name = os.path.dirname(dir_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
    
    return paths

class DataProcessor:
    """Process crystal structure data for CHGnet."""

    def __init__(self, config: dict):
        """
        Initialize data processor.

        Args:
            config (dict): Configuration parameters.
        """
        self.structures_dir = Path(config['structures_dir'])
        self.file_path = Path(config['file_path'])
        self.batch_size = config.get('batch_size', 32)
        self.split_ratio = config.get('split_ratio', [0.7, 0.1, 0.2])
        self.random_state = config.get('random_state', 42)
        self.max_samples = config.get('max_samples', None)
        
        # Essential data containers
        self.structures: List[Structure] = []
        self.energies: List[float] = []
        self.dataset = None
    
    def read_poscar(self, file_path: str) -> Optional[Structure]:
        """Read structure from POSCAR file."""
        poscar = Poscar.from_file(file_path)
        return poscar.structure

    def load_data(self) -> None:
        """Load structures and energy per atom from CSV file."""
        print("Loading data...")
        df = pd.read_csv(self.file_path)
        
        if self.max_samples is not None and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=self.random_state)
        
        for _, row in df.iterrows():
            try:
                struct_path = os.path.join(self.structures_dir, row['FileName'])
                struct = self.read_poscar(struct_path)
                if struct is None:
                    continue
                
                energy_per_atom = float(row['Energy_per_atom_eV'])
                
                # Basic validation
                if np.isnan(energy_per_atom) or np.isinf(energy_per_atom):
                    print(f"Invalid energy for {row['FileName']}, skipping.")
                    continue
                
                self.structures.append(struct)
                self.energies.append(energy_per_atom)
                
            except Exception as e:
                print(f"Error in {row['FileName']}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(self.structures)} structures.")

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for training, validation, and testing."""
        self.dataset = StructureData(
            structures=self.structures,
            energies=self.energies,
            forces=None,
            stresses=None,
            magmoms=None
        )
        
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            train_ratio=self.split_ratio[0],
            val_ratio=self.split_ratio[1],
        )
        
        print(f"\nDataset split:")
        print(f"Train: {len(train_loader.dataset)} samples")
        print(f"Validation: {len(val_loader.dataset)} samples")
        print(f"Test: {len(test_loader.dataset)} samples")
        
        return train_loader, val_loader, test_loader