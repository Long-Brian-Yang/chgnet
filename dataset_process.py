from __future__ import annotations
import os
import warnings
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
from pymatgen.core import Structure 
from pymatgen.io.vasp import Poscar
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader
from dgl.data.utils import split_dataset

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
    """Process crystal structure data for M3GNet."""
    
    def __init__(self, config: dict):
        """
        Initialize data processor.
        
        Args:
            config (dict): Configuration containing:
                - structures_dir: Path to structure files
                - file_path: Path to data list
                - cutoff: Cutoff radius for graphs
                - batch_size: Batch size for loading
                - split_ratio: Train/val/test split
                - random_state: Random seed
        """
        self.structures_dir = Path(config['structures_dir'])
        self.file_path = Path(config['file_path'])
        self.cutoff = config.get('cutoff', 4.0)
        self.batch_size = config.get('batch_size', 32)
        self.split_ratio = config.get('split_ratio', [0.7, 0.1, 0.2])
        self.random_state = config.get('random_state', 42)
        
        # Essential data containers
        self.structures: List[Structure] = []
        self.bandgap_values: List[float] = []
        self.dataset: Optional[MGLDataset] = None
        self.element_list: List[str] = []
    
    def read_poscar(self, file_path: str) -> Structure:
        """Read POSCAR file."""
        poscar = Poscar.from_file(file_path)
        return poscar.structure
    
    def load_data(self, bandgap_column: str = 'Bandgap_by_DFT') -> None:
        """Load structures and bandgap values."""
        print("Loading data from files...")
        df = pd.read_csv(self.file_path)
        sampled_df = df.sample(frac=1.0, random_state=self.random_state)
        
        for _, row in sampled_df.iterrows():
            try:
                file_name = row['FileName']
                struct = self.read_poscar(
                    os.path.join(self.structures_dir, file_name)
                )
                band_v = row[bandgap_column]
                
                self.structures.append(struct)
                self.bandgap_values.append(band_v)
                
            except Exception as e:
                print(f"Error processing file {row['FileName']}: {str(e)}")
        
        print(f"Successfully loaded {len(self.structures)} structures")
    
    def create_dataset(self, normalize: bool = False) -> MGLDataset:
        """Create graph dataset."""
        if not self.structures:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get unique elements
        self.element_list = get_element_list(self.structures)
        
        # Initialize graph converter
        converter = Structure2Graph(
            element_types=self.element_list,
            cutoff=self.cutoff
        )
        
        # Create dataset - 使用示例代码中的方式
        self.dataset = MGLDataset(
            structures=self.structures,
            converter=converter,
            labels={"bandgap": self.bandgap_values}
        )
        
        return self.dataset
    
    def create_dataloaders(self) -> Tuple[MGLDataLoader, MGLDataLoader, MGLDataLoader]:
        """Create train, validation and test dataloaders."""
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        # Split dataset
        train_data, val_data, test_data = split_dataset(
            self.dataset,
            frac_list=self.split_ratio,
            shuffle=True,
            random_state=self.random_state
        )

        # num_workers = 1

        # Create data loaders using MGLDataLoader
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True
        )
        
        print(f"Created dataloaders - Train: {len(train_data)}, "
              f"Val: {len(val_data)}, Test: {len(test_data)} samples")
        
        return train_loader, val_loader, test_loader

def main():
    """Test data processing pipeline."""
    paths = get_project_paths()
    
    config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'cutoff': 4.0,
        'batch_size': 16,
        'split_ratio': [0.5, 0.1, 0.4],
        'random_state': 42
    }
    
    processor = DataProcessor(config)
    processor.load_data()
    dataset = processor.create_dataset()
    train_loader, val_loader, test_loader = processor.create_dataloaders()
    
    print("Data processing completed successfully!")

if __name__ == "__main__":
    main()