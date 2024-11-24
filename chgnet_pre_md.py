from __future__ import annotations
import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from ase import Atoms
from ase.io import read, write, Trajectory
from ase.io.vasp import read_vasp
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
from ase import Atom
import matgl
from matgl.ext.ase import PESCalculator
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from dataset_process import get_project_paths, DataProcessor

class MDSystem:
    """Run molecular dynamics simulations using M3GNet potential."""
    
    def __init__(
        self,
        config: dict,
        model_name: str = "M3GNet-MP-2021.2.8-PES",
        time_step: float = 1.0,  # fs
        friction: float = 0.02,
        total_steps: int = 100,
        output_interval: int = 100
    ):
        # Get paths and setup directories
        self.paths = get_project_paths()
        self.structures_dir = Path(self.paths['structures_dir'])
        self.output_dir = Path(self.paths['output_dir'])
        self.md_output_dir = self.output_dir / 'md_trajectories'
        
        # Configuration
        self.config = config
        self.time_step = time_step
        self.friction = friction
        self.total_steps = total_steps
        self.output_interval = output_interval
        
        # Setup environment and logging
        self.setup_environment()
        
        # Initialize potential and calculator
        self.potential = matgl.load_model(model_name)
        self.calculator = PESCalculator(self.potential)
        self.logger.info(f"Loaded potential model: {model_name}")
        
        # Initialize data processor
        self.data_processor = DataProcessor(config)
        
        # Structure handlers
        self.atoms_adaptor = AseAtomsAdaptor()
    
    def setup_environment(self):
        """Setup logging and directories."""
        os.makedirs(self.md_output_dir, exist_ok=True)
        
        # Setup logging
        log_file = self.md_output_dir / f"md_simulation_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("main")
        self.logger.info(f"Working directory: {self.md_output_dir}")
        self.logger.info(f"Structures directory: {self.structures_dir}")
    
    def find_vasp_files(self) -> List[Path]:
        """Find all .vasp files in structures directory."""
        vasp_files = list(self.structures_dir.glob("**/*.vasp"))
        if not vasp_files:
            self.logger.warning(f"No .vasp files found in {self.structures_dir}")
        else:
            self.logger.info(f"Found {len(vasp_files)} .vasp files")
            for f in vasp_files:
                self.logger.info(f"Found structure file: {f}")
        return vasp_files
    
    def add_protons(self, atoms: Atoms, n_protons: int) -> Atoms:
        """
        Add protons to the structure based on theoretical understanding.
        
        References:
        1. Kreuer, K. D., Solid State Ionics (1999)
        2. Björketun, M. E., et al., PRB (2005)
        3. Gomez, M. A., et al., SSI (2010)
        """
        # Find oxygen atoms
        o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols())
                    if symbol == 'O']
        
        if len(o_indices) < n_protons:
            self.logger.warning(f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
            n_protons = len(o_indices)
        
        # Theoretical OH bond length (from Gomez et al.)
        OH_BOND_LENGTH = 0.98  # Å
        
        # Add protons near selected oxygen atoms
        used_oxygens = []
        for i in range(n_protons):
            # Select oxygen atom that hasn't been used
            available_oxygens = [idx for idx in o_indices if idx not in used_oxygens]
            if not available_oxygens:
                self.logger.warning("No more available oxygen atoms for proton incorporation")
                break
                
            o_idx = available_oxygens[0]
            used_oxygens.append(o_idx)
            
            o_pos = atoms.positions[o_idx]
            
            # Find neighboring oxygen atoms to determine optimal proton position
            neighbors = []
            for other_idx in o_indices:
                if other_idx != o_idx:
                    dist = atoms.get_distance(o_idx, other_idx)
                    if dist < 3.0:  # Consider O atoms within 3 Å
                        neighbors.append(atoms.positions[other_idx])
            
            # Calculate optimal proton position
            if neighbors:
                # Avoid positioning proton towards other oxygen atoms
                direction = np.zeros(3)
                for neighbor in neighbors:
                    vec = o_pos - neighbor
                    vec = vec / np.linalg.norm(vec)
                    direction += vec
                
                if np.any(direction):
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([1, 0, 0])  # Default direction if no clear preference
            else:
                direction = np.array([1, 0, 0])  # Default direction if no neighbors
                
            # Calculate proton position
            h_pos = o_pos + direction * OH_BOND_LENGTH
            
            # Add proton
            atoms.append(Atom('H', position=h_pos))
            self.logger.info(f"Added proton near O atom {o_idx} at position: {h_pos}")
            
            # Log OH bond length for verification
            oh_dist = np.linalg.norm(h_pos - o_pos)
            self.logger.info(f"Created OH bond with length: {oh_dist:.3f} Å")
        
        self.logger.info(f"Added {n_protons} protons forming OH groups")
        self.logger.info(f"New composition: {atoms.get_chemical_formula()}")
        
        return atoms
    
    def run_md(
        self,
        structure_file: Path,
        temperature: float,
        traj_file: Optional[Path] = None
    ) -> str:
        """Run MD simulation for given structure and temperature."""
        # Create output directory for this structure
        struct_name = structure_file.stem
        struct_output_dir = self.md_output_dir / struct_name
        os.makedirs(struct_output_dir, exist_ok=True)
        
        # Read structure using ASE's VASP reader
        try:
            atoms = read_vasp(str(structure_file))
            atoms.calc = self.calculator
            
            self.logger.info(f"Loaded structure from {structure_file}")
            self.logger.info(f"Structure composition: {atoms.get_chemical_formula()}")
            
        except Exception as e:
            self.logger.error(f"Error loading structure from {structure_file}: {e}")
            raise
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        # Setup Langevin dynamics
        dyn = Langevin(
            atoms,
            timestep=self.time_step * fs,
            temperature_K=temperature,
            friction=self.friction
        )
        
        # Setup trajectory file
        if traj_file is None:
            traj_file = struct_output_dir / f"MD_{int(temperature):04d}.traj"
        traj = Trajectory(str(traj_file), 'w', atoms)
        dyn.attach(traj.write, interval=self.output_interval)
        
        # Run MD
        self.logger.info(f"Starting MD at {temperature}K for {struct_name}")
        self.logger.info(f"Total steps: {self.total_steps}, Output interval: {self.output_interval}")
        
        for step in range(1, self.total_steps + 1):
            dyn.run(1)
            if step % 1000 == 0:
                temp = atoms.get_temperature()
                self.logger.info(f"Step {step}/{self.total_steps}, Temperature: {temp:.1f}K")
                
                # Save current state as VASP format
                checkpoint_file = struct_output_dir / f"POSCAR_checkpoint_{step}"
                write(str(checkpoint_file), atoms, format='vasp')
        
        self.logger.info(f"MD simulation completed. Trajectory saved to {traj_file}")
        return str(traj_file)
    
    def run_temperature_range(
        self,
        structure_files: List[Path] = None,
        temperatures: List[float] = None
    ) -> Dict[str, Dict[float, str]]:
        """Run MD simulations for multiple structures at different temperatures."""
        if structure_files is None:
            structure_files = self.find_vasp_files()
        
        if not structure_files:
            self.logger.error("No structure files found")
            return {}
            
        if temperatures is None:
            temperatures = [800, 900, 1000]  # default temperatures
        
        results = {}
        for struct_file in structure_files:
            struct_name = struct_file.stem
            results[struct_name] = {}
            
            for temp in temperatures:
                try:
                    traj_file = self.run_md(struct_file, temp)
                    results[struct_name][temp] = traj_file
                except Exception as e:
                    self.logger.error(f"Error running MD for {struct_name} at {temp}K: {str(e)}")
        
        return results

# def main():
#     """Main function to run MD simulations."""
#     # Get project paths
#     paths = get_project_paths()
    
#     # Setup configuration
#     config = {
#         'structures_dir': paths['structures_dir'],
#         'file_path': paths['file_path'],
#         'cutoff': 4.0,
#         'batch_size': 16,
#         'split_ratio': [0.5, 0.1, 0.4],
#         'random_state': 42
#     }
    
#     # Initialize MD system
#     md_system = MDSystem(
#         config=config,
#         time_step=1.0,
#         friction=0.02,
#         total_steps=20000,
#         output_interval=100
#     )
    
#     # Define temperatures
#     temperatures = [800, 900, 1000]  # K
    
#     # Run MD simulations for all .vasp files
#     print("\nRunning MD simulations...")
#     results = md_system.run_temperature_range(temperatures=temperatures)
    
#     print("\nCompleted MD simulations:")
#     for struct_name, temp_dict in results.items():
#         print(f"\nStructure: {struct_name}")
#         for temp, traj_file in temp_dict.items():
#             print(f"  {temp}K: {traj_file}")

# if __name__ == "__main__":
#     main()

def main():
    """Test MD simulation with specific structure."""
    # Get project paths
    paths = get_project_paths()
    
    # Setup configuration
    config = {
        'structures_dir': paths['structures_dir'],
        'file_path': paths['file_path'],
        'cutoff': 4.0,
        'batch_size': 16,
        'split_ratio': [0.5, 0.1, 0.4],
        'random_state': 42
    }
    
    # Initialize MD system
    md_system = MDSystem(
        config=config,
        time_step=1.0,
        friction=0.02,
        total_steps=50,  # 50 steps for testing
        output_interval=10
    )
    
    # Test with specific structure
    structure_file = Path("data/Ba8Zr8O24.vasp")
        
    print(f"\nTesting with structure: {structure_file}")
    
    # Test with single temperature
    temperatures = [500, 700, 1000]  # K
    
    try:
        # 1. 读取初始结构
        atoms = read_vasp(str(structure_file))
        print(f"\nLoaded initial structure: {atoms.get_chemical_formula()}")
        
        # 2. 添加质子
        n_protons = 8  # 添加8个质子
        atoms = md_system.add_protons(atoms, n_protons)
        
        # 3. 保存掺H结构
        hydrated_file = md_system.md_output_dir / f"{structure_file.stem}_H{n_protons}.vasp"
        write(str(hydrated_file), atoms, format='vasp')
        print(f"Saved hydrated structure to: {hydrated_file}")
        
        # 4. 运行不同温度的MD模拟
        temperatures = [500, 700, 1000]  # K
        trajectories = {}
        
        for temp in temperatures:
            print(f"\nRunning MD at {temp}K...")
            
            # 创建特定温度的输出目录
            temp_dir = md_system.md_output_dir / f"T_{temp}K"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 定义轨迹文件路径
            traj_file = temp_dir / f"MD_{temp}K.traj"
            
            try:
                # 运行MD
                traj_path = md_system.run_md(
                    structure_file=hydrated_file,
                    temperature=temp,
                    traj_file=traj_file
                )
                
                trajectories[temp] = traj_path
                print(f"Completed MD at {temp}K")
                print(f"Trajectory saved to: {traj_path}")
                
                # 保存最后一帧的结构
                final_structure = temp_dir / f"FINAL_POSCAR_{temp}K"
                final_atoms = read(traj_path, index=-1)
                write(str(final_structure), final_atoms, format='vasp')
                print(f"Final structure saved to: {final_structure}")
                
            except Exception as e:
                print(f"Error running MD at {temp}K: {str(e)}")
                continue
        
        # 5. 打印总结
        print("\nMD Simulation Summary:")
        print("-" * 50)
        print(f"Initial structure: {structure_file}")
        print(f"Protons added: {n_protons}")
        print(f"Temperatures simulated: {temperatures} K")
        print("\nTrajectory files:")
        for temp, traj in trajectories.items():
            print(f"  {temp}K: {traj}")
        
    except Exception as e:
        print(f"\nError in MD simulation: {str(e)}")
        raise
        
    finally:
        # 保存配置信息
        config_info = {
            'structure_file': str(structure_file),
            'n_protons': n_protons,
            'temperatures': temperatures,
            'md_parameters': {
                'time_step': md_system.time_step,
                'friction': md_system.friction,
                'total_steps': md_system.total_steps,
                'output_interval': md_system.output_interval
            }
        }
        
        import json
        config_file = md_system.md_output_dir / 'md_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_info, f, indent=4)
        print(f"\nConfiguration saved to: {config_file}")

if __name__ == "__main__":
    main()
