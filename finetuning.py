# finetune.py

from __future__ import annotations
import os
import json
import time
import logging
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader


from dataset_process import DataProcessor, get_project_paths

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

class FineTuner:
    def __init__(
        self,
        working_dir: str,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize the FineTuner.

        Args:
            working_dir (str): Directory where outputs will be saved.
            debug (bool): If True, sets logging level to DEBUG.
            **kwargs: Additional configuration parameters.
        """
        self.working_dir = Path(working_dir)
        
        # Define output directories
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.logs_dir = self.working_dir / "logs"
        self.results_dir = self.working_dir / "results"
        
        # Create directories
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.debug = debug
        self.model = None
        self.best_model = None
        
        # Training configuration
        self.config = {
            'batch_size': kwargs.get('batch_size', 32),
            'num_epochs': kwargs.get('num_epochs', 5),
            'learning_rate': kwargs.get('learning_rate', 1e-3),
            'split_ratio': kwargs.get('split_ratio', [0.7, 0.1, 0.2]),
            'random_state': kwargs.get('random_state', 42),
            'max_samples': kwargs.get('max_samples', None),
            'frozen_layers': kwargs.get('frozen_layers', []),
            'train_composition': kwargs.get('train_composition', False)
        }
        
        # Setup logging and save configuration
        self._setup_logging()
        self._save_config()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / 'train.log'
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _save_config(self):
        """Save configuration to JSON file."""
        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def freeze_layers(self):
        """Freeze specified layers in the model."""
        layer_map = {
            'atom_embedding': self.model.atom_embedding,
            'bond_embedding': self.model.bond_embedding,
            'angle_embedding': self.model.angle_embedding,
            'bond_expansion': self.model.bond_basis_expansion,
            'angle_expansion': self.model.angle_basis_expansion,
            'atom_conv': self.model.atom_conv_layers[:-1],
            'bond_conv': self.model.bond_conv_layers,
            'angle_layers': self.model.angle_layers,
        }
        
        for layer_name in self.config['frozen_layers']:
            if layer_name in layer_map:
                layer = layer_map[layer_name]
                if isinstance(layer, (list, tuple)):
                    for sublayer in layer:
                        for param in sublayer.parameters():
                            param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
                self.logger.info(f"Frozen layer: {layer_name}")
    
    def train(self):
        """Execute the training process."""
        self.logger.info("Starting fine-tuning process...")
        start_time = time.time()
        
        try:
            # 1. Prepare data
            paths = get_project_paths()
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'batch_size': self.config['batch_size'],
                'split_ratio': self.config['split_ratio'],
                'random_state': self.config['random_state'],
                'max_samples': self.config['max_samples']
            }
            
            processor = DataProcessor(data_config)
            processor.load_data()
            train_loader, val_loader, test_loader = processor.create_dataloaders()
            
            # Load and configure model
            self.model = CHGNet.load()
            self.freeze_layers()
            self.logger.info("Loaded and configured CHGNet model")
            
            # Setup trainer
            trainer = Trainer(
                model=self.model,
                targets="e",  # Only training on energy
                optimizer="Adam",
                scheduler="CosLR",
                criterion="MSE",
                epochs=self.config['num_epochs'],
                learning_rate=self.config['learning_rate'],
                use_device="cpu",
                print_freq=10
            )
            
            # Train model
            self.logger.info("Starting training...")
            trainer.train(
                train_loader,
                val_loader,
                test_loader,
                train_composition_model=self.config['train_composition']
            )
            
            # Save models
            self.model = trainer.model
            self.best_model = trainer.best_model
            
            model_path = self.checkpoints_dir / 'final_model'
            best_model_path = self.checkpoints_dir / 'best_model'
            
            self.model.save(str(model_path))
            self.best_model.save(str(best_model_path))
            
            self.logger.info(f"Models saved to {self.checkpoints_dir}")
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return self.best_model
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def plot_training_curves(self, metrics):
        """Plot and save training curves."""
        try:
            # Load training metrics
            metrics_file = self.logs_dir / "metrics.csv"
            if not metrics_file.exists():
                self.logger.warning("Metrics file not found. Skipping plotting.")
                return
            
            metrics = pd.read_csv(metrics_file)

            plt.figure(figsize=(10, 6))
            
            if "train_MAE" in metrics.columns:
                metrics["train_MAE"].dropna().plot(label='Training MAE')
            if "val_MAE" in metrics.columns:
                metrics["val_MAE"].dropna().plot(label='Validation MAE')
            
            plt.xlabel('Epochs')
            # plt.ylabel('MAE')
            plt.ylabel('Energy MAE (eV/atom)')
            plt.legend()
            plt.savefig(self.logs_dir / "training_curve.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting training curves: {str(e)}")

def main():
    """Main function to start fine-tuning."""
    paths = get_project_paths()
    
    trainer = FineTuner(
        working_dir=paths['output_dir'],
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=32,
        split_ratio=[0.7, 0.1, 0.2],
        random_state=42,
        max_samples=None,
        frozen_layers=['atom_embedding', 'bond_embedding', 'angle_embedding'],
        train_composition=True,
        debug=True
    )
    
    best_model = trainer.train()
    trainer.plot_training_curves()
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()