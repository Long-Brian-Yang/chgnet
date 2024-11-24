from __future__ import annotations

import os
import logging
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from scipy import stats
from typing import List, Dict, Optional
from dataset_process import DataProcessor, get_project_paths
import matgl
from matgl.ext.ase import PESCalculator
from concurrent.futures import ProcessPoolExecutor

class MDSimulator:
    """Run MD simulations using MatGL's M3GNet potential."""
    
    def __init__(
        self,
        working_dir: Path,
        model_name: str = "M3GNet-MP-2021.2.8-PES",
        time_step: float = 1.0,  # fs
        friction: float = 0.02,
        total_steps: int = 10000,
        output_interval: int = 100
    ):
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.time_step = time_step
        self.friction = friction
        self.total_steps = total_steps
        self.output_interval = output_interval
        
        # Setup logging
        log_file = self.working_dir / f"md_simulation_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filename=log_file
        )
        self.logger = logging.getLogger()
        
        # Load MatGL M3GNet potential
        self.potential = matgl.load_model(model_name)
        self.calculator = PESCalculator(self.potential)
        self.logger.info(f"Loaded potential model: {model_name}")
    
    def run_md(
        self,
        structure_file: str,
        temperature: float,
        traj_file: Optional[str] = None
    ):
        # Read structure
        atoms = read(structure_file)
        atoms.calc = self.calculator
        # atoms.set_calculator(self.calculator)
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        # Setup Langevin dynamics
        dyn = Langevin(
            atoms,
            timestep=self.time_step * fs,
            temperature_K=temperature,  # Temperature in Kelvin
            friction=self.friction
        )
        
        # Setup trajectory
        if traj_file is None:
            traj_file = self.working_dir / f"MD_{int(temperature)}K.traj"
        traj = Trajectory(traj_file, 'w', atoms)
        dyn.attach(traj.write, interval=self.output_interval)
        
        self.logger.info(f"Starting MD at {temperature}K")
        for step in range(1, self.total_steps + 1):
            dyn.run(1)
            if step % 1000 == 0:
                temp = atoms.get_temperature()
                self.logger.info(f"Step {step}/{self.total_steps}, Temperature: {temp:.1f}K")
        
        self.logger.info(f"MD simulation completed. Trajectory saved to {traj_file}")

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    config = load_config('config.json')
    paths = get_project_paths()
    # Define paths
    working_dir = Path("md_output")
    structure_file = Path(paths['structures_dir'])
    structure_files = list(structure_file.glob('**/*.vasp'))

    # Initialize simulator
    simulator = MDSimulator(
        working_dir=working_dir,
        model_name=config['model_name'],
        time_step=config['time_step'],
        friction=config['friction'],
        total_steps=config['total_steps'],
        output_interval=config['output_interval']
    )
    
    # Define temperatures
    temperatures = config['temperatures']
    
    # Run simulations
    with ProcessPoolExecutor() as executor:
        for temp in temperatures:
            for structure_file in structure_files:
                simulator.run_md(structure_file=str(structure_file), temperature=temp)

if __name__ == "__main__":
    main()