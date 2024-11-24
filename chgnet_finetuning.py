from __future__ import annotations
import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pymatgen.core import Structure
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
        Initialize the FineTuner for CHGNet fine-tuning.

        Args:
            working_dir (str): Directory where all outputs (logs, checkpoints, results) will be saved.
            debug (bool, optional): If True, sets logging level to DEBUG. Defaults to False.
            **kwargs: Additional configuration parameters such as batch_size, epochs, etc.
        """

        self.working_dir = Path(working_dir)

        # Define directories for checkpoints, logs, and results
        self.checkpoints_dir = self.working_dir / "checkpoints"
        self.logs_dir = self.working_dir / "logs"
        self.results_dir = self.working_dir / "results"

        # Create directories if they do not exist
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.debug = debug
        self.model = None
        self.trainer = None

        # Training configuration with default values
        self.config = {
            'batch_size': kwargs.get('batch_size', 8),
            'epochs': kwargs.get('epochs', 5),
            'learning_rate': kwargs.get('learning_rate', 1e-2),
            'use_device': kwargs.get('use_device', 'cpu'),
            'train_ratio': kwargs.get('train_ratio', 0.8),
            'val_ratio': kwargs.get('val_ratio', 0.1),
            'random_state': kwargs.get('random_state', 42),
            'freeze_layers': kwargs.get('freeze_layers', True),
            'train_composition_model': kwargs.get('train_composition_model', False)
        }

        # Setup logging and save configuration
        self._setup_logging()
        self._save_config()
    
    def _setup_logging(self):
        """
        Setup logging configuration to log messages to both a file and the console.
        """

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
        """
        Save the training configuration to a JSON file for future reference.
        """

        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_model(self):
        """
        Setup the CHGNet model.
        """

        try:
            self.logger.info("Setting up CHGNet model...")
            
            # Load pretrained CHGNet
            self.model = CHGNet.load()
            self.logger.info("CHGNet model loaded successfully")

            # Optionally freeze certain layers
            if self.config['freeze_layers']:
                self.logger.info("Freezing specified layers in the model")
                for layer in [
                    self.model.atom_embedding,
                    self.model.bond_embedding,
                    self.model.angle_embedding,
                    self.model.bond_basis_expansion,
                    self.model.angle_basis_expansion,
                    self.model.atom_conv_layers[:-1],
                    self.model.bond_conv_layers,
                    self.model.angle_layers,
                ]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                self.logger.info("Not freezing any layers in the model")
            
        except Exception as e:
            self.logger.error(f"Error in model setup: {str(e)}")
            raise

    def run_training(self, data_paths: Dict):
        """
        Execute the fine-tuning process for CHGNet.

        Args:
            data_paths (Dict): Dictionary containing paths to data files.
        """

        start_time = time.time()
        self.logger.info("Starting training process...")

        try:
            # 1. Prepare data configuration
            data_config = {
                'structures_dir': paths['structures_dir'],
                'file_path': paths['file_path'],
                'batch_size': self.config['batch_size'],
                'train_ratio': self.config['train_ratio'],
                'val_ratio': self.config['val_ratio'],
                'random_state': self.config['random_state']
            }

            # Create StructureData dataset
            dataset = StructureData(
                structures=structures,
                energies=energies,
                forces=forces,
                stresses=stresses,
                magmoms=magmoms,
            )

            # Split dataset into train, val, test loaders
            train_loader, val_loader, test_loader = get_train_val_test_loader(
                dataset,
                batch_size=self.config['batch_size'],
                train_ratio=self.config['train_ratio'],
                val_ratio=self.config['val_ratio'],
                random_seed=self.config['random_state']
            )

            # 2. Setup model
            self.setup_model()

            # 3. Define Trainer
            self.trainer = Trainer(
                model=self.model,
                targets="efsm",
                optimizer="Adam",
                scheduler="CosLR",
                criterion="MSE",
                epochs=self.config['epochs'],
                learning_rate=self.config['learning_rate'],
                use_device=self.config['use_device'],
                print_freq=10,
            )

            # 4. Start training
            self.logger.info("Starting training...")
            self.trainer.train(
                train_loader,
                val_loader,
                test_loader,
                train_composition_model=self.config['train_composition_model']
            )

            # 5. Save the trained model
            model_save_path = self.checkpoints_dir / 'trained_chgnet.pth'
            torch.save(self.trainer.model.state_dict(), model_save_path)
            self.logger.info(f"Model saved to {model_save_path}")

            # 6. Extract and save training metrics
            # Since CHGNet's Trainer does not log to files by default, we'll collect metrics manually
            metrics = self.trainer.metrics  # Assuming the trainer collects metrics in a dictionary
            metrics_file = self.results_dir / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Training metrics saved to {metrics_file}")

            # 7. Plot training curves
            self.plot_training_curves(metrics)

            duration = time.time() - start_time
            self.logger.info(f"Training completed in {duration:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def plot_training_curves(self, metrics):
        """
        Plot and save the training and validation MAE curves.

        Args:
            metrics (dict): Dictionary containing training metrics.
        """
        try:
            plt.figure(figsize=(10, 6))

            epochs = range(len(metrics['train_e_MAE']))

            # Plot Training MAE
            plt.plot(epochs, metrics['train_e_MAE'], label='Training Energy MAE')
            plt.plot(epochs, metrics['val_e_MAE'], label='Validation Energy MAE')

            plt.xlabel('Epochs')
            plt.ylabel('Energy MAE')
            plt.legend()

            plot_path = self.logs_dir / "training_curve_energy.png"
            plt.savefig(
                plot_path,
                facecolor='w',
                bbox_inches="tight",
                pad_inches=0.3,
                transparent=True
            )
            plt.close()

            # Similarly, you can plot force, stress, and magmom MAEs if available

        except Exception as e:
            self.logger.error(f"Error plotting training curves: {str(e)}")

def main():
    """
    Main function to initiate the fine-tuning process.
    """

    # Define data paths
    data_paths = {
        'dataset_file': 'path_to_your_dataset/chgnet_dataset.json'
    }

    # Initialize the FineTuner with the working directory and configuration parameters
    trainer = FineTuner(
        working_dir='output_dir',
        epochs=5,
        learning_rate=1e-2,
        batch_size=8,
        train_ratio=0.8,
        val_ratio=0.1,
        random_state=42,
        use_device='cpu',
        freeze_layers=True,
        train_composition_model=False
    )
    
    trainer.run_training(data_paths)

if __name__ == "__main__":
    main()
