from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from datetime import datetime
from scipy import stats
from ase.io import read
from ase.units import kB
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib
from dataset_process import get_project_paths
import warnings

warnings.simplefilter("ignore")

matplotlib.use('Agg')


class MDAnalysisSystem:
    """
    Complete system for analyzing molecular dynamics trajectories.
    Integrates with pymatgen for structure handling.
    """

    def __init__(
        self,
        config: dict,
        structure_name: str,
        target_atom: str = 'H',
        time_step: float = 1.0,  # fs
        shift_time: int = 50,   # time steps
        window_size: int = 100   # time steps
    ):
        self.paths = get_project_paths()
        self.structures_dir = Path(self.paths['structures_dir'])
        self.output_dir = Path(self.paths['output_dir'])
        self.working_dir = self.output_dir / 'md_analysis'

        self.structure_name = structure_name
        self.target_atom = target_atom
        self.time_step = time_step
        self.shift_time = shift_time
        self.window_size = window_size
        self.config = config

        # Setup environment
        self.setup_environment()
        self.setup_plot_parameters()

        # Initialize structure handler
        self.atoms_adaptor = AseAtomsAdaptor()

    def setup_environment(self):
        """Setup logging and directories."""
        os.makedirs(self.working_dir, exist_ok=True)
        self.plot_dir = self.working_dir / 'plots'
        os.makedirs(self.plot_dir, exist_ok=True)

        # Setup logging
        log_file = self.working_dir / \
            f"md_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("main")

    def setup_plot_parameters(self):
        """Setup matplotlib parameters."""
        self.font_size = 26
        params = {
            'axes.labelsize': self.font_size,
            'font.size': self.font_size,
            'font.family': 'DejaVu Sans',
            'legend.fontsize': self.font_size,
            'xtick.labelsize': self.font_size,
            'ytick.labelsize': self.font_size,
            'axes.titlesize': self.font_size,
            'text.usetex': False,
            'figure.figsize': [14, 14]
        }
        plt.rcParams.update(params)

    def analyze_single_trajectory(self, temperature: int) -> dict:
        """Analyze a single MD trajectory."""
        # Find trajectory file in temperature directory
        temp_dir = self.working_dir / f"T_{temperature}K"
        traj_file = temp_dir / f"MD_{temperature}K.traj"

        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

        # Read trajectory
        traj_list = read(traj_file, index=":")
        n_frames = len(traj_list)

        # Adjust window size based on available frames
        adjusted_window_size = min(self.window_size, n_frames - 1)
        adjusted_shift = min(self.shift_time, adjusted_window_size // 2)

        if adjusted_window_size < 3:  # Need at least 3 frames for MSD
            self.logger.warning(f"Not enough frames ({n_frames}) for analysis")
            return None

        # Get target atom indices and volume
        atom_indices = [
            i for i, x in enumerate(
                traj_list[0].get_chemical_symbols()) if x == self.target_atom]

        volume = np.mean([atoms.get_volume() for atoms in traj_list])
        volume_cm3 = volume * 1e-24  # Convert to cm³

        self.logger.info(f"Analyzing trajectory at {temperature}K")
        self.logger.info(
            f"Number of {self.target_atom} atoms: {len(atom_indices)}")

        try:
            # Extract positions for target atoms
            positions = np.array([traj.get_positions() for traj in traj_list])
            positions = positions[:, atom_indices]

            self.logger.info(f"Position array shape: {positions.shape}")

            # Calculate number of complete windows
            n_windows = max(
                1, (n_frames - adjusted_window_size) // adjusted_shift + 1)

            self.logger.info(f"Number of analysis windows: {n_windows}")

            msd_windows = []
            diffusion_coefficients = []
            # fixed time array(fs->ps)
            time_array = np.arange(adjusted_window_size) * \
                self.time_step / 1000

            for i in range(n_windows):
                start_idx = i * adjusted_shift
                end_idx = start_idx + adjusted_window_size

                if end_idx > len(positions):
                    break

                # Calculate MSD for this window
                window_positions = positions[start_idx:end_idx]
                ref_positions = window_positions[0]
                displacements = window_positions - \
                    ref_positions[np.newaxis, :, :]

                squared_displacements = displacements**2

                # Calculate MSD components
                msd_components = {
                    'x': np.mean(squared_displacements[..., 0], axis=1),
                    'y': np.mean(squared_displacements[..., 1], axis=1),
                    'z': np.mean(squared_displacements[..., 2], axis=1)
                }
                msd_total = np.mean(
                    np.sum(
                        squared_displacements,
                        axis=2),
                    axis=1)

                # Store results
                msd_windows.append(msd_total)

                # Calculate diffusion coefficient
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_array, msd_total)
                D = slope / 6  # Einstein relation in Å²/ps
                D_cm2_s = D * 1e-4  # Convert to cm²/s (1e-16/1e-12 = 1e-4)
                diffusion_coefficients.append(D_cm2_s)

                # Plot first window
                if i == 0:
                    msd_components['total'] = msd_total
                    self.plot_msd_components(
                        time_array, msd_components, temperature,
                        f"msd_components_{temperature}K.png"
                    )

            # Calculate averages and statistics
            avg_msd = np.mean(msd_windows, axis=0)
            avg_D = np.mean(diffusion_coefficients)
            std_D = np.std(diffusion_coefficients)

            # Calculate conductivity
            conductivity = self.calculate_conductivity(
                avg_D, temperature, volume_cm3, len(atom_indices)
            )

            # Plot average MSD
            self.plot_average_msd(
                time_array, avg_msd, avg_D, temperature,
                f"average_msd_{temperature}K.png"
            )

            results = {
                'temperature': temperature,
                'T_inverse': 1000 / temperature,
                'D_cm2_s': avg_D,
                'D_std': std_D,
                'log10_D': np.log10(avg_D),
                'conductivity': conductivity,
                'log10_conductivity': np.log10(conductivity),
                'volume_cm3': volume_cm3,
                'n_carriers': len(atom_indices),
                'msd_A2ps': avg_D * 6 * 1e4  # Convert back to Å²/ps for reporting
            }

            self.logger.info(f"Results for {temperature}K:")
            self.logger.info(f"MSD: {results['msd_A2ps']:.4f} Å²/ps")
            self.logger.info(
                f"Diffusion coefficient: {avg_D:.2e} ± {std_D:.2e} cm²/s")
            self.logger.info(f"Conductivity: {conductivity:.2e} S/cm")

            return results

        except Exception as e:
            self.logger.error(f"Error in MSD calculation: {str(e)}")
            self.logger.error(f"Positions shape: {positions.shape}")
            self.logger.error(f"Time array shape: {time_array.shape}")
            raise

    def calculate_conductivity(self, D_cm2_s: float, temperature: float,
                               volume_cm3: float, n_carriers: int) -> float:
        """Calculate ionic conductivity using the Nernst-Einstein relation."""
        # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7761793/
        # σ = (N*e²*D)/(k_B*T)
        e = 1.602176634e-19  # Elementary charge (C)
        kb = 1.38064852e-23    # Boltzmann constant (J/K)
        N = n_carriers / volume_cm3  # Carrier density (cm⁻³)

        sigma = (N * e * e * D_cm2_s) / (kb * temperature)  # S/cm
        return sigma

    def analyze_temperature_range(
            self, temperatures: list = None) -> pd.DataFrame:
        """Analyze MD trajectories for a range of temperatures."""
        if temperatures is None:
            # Find all trajectory files
            traj_files = self.find_trajectory_files()
            temperatures = []
            for f in traj_files:
                try:
                    temp = int(f.stem.split('_')[-1])
                    temperatures.append(temp)
                except ValueError:
                    self.logger.warning(
                        f"Could not extract temperature from {f}")

        if not temperatures:
            self.logger.error("No temperatures to analyze")
            return pd.DataFrame()

        results = []
        for temp in sorted(temperatures):
            try:
                result = self.analyze_single_trajectory(temp)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing {temp}K: {str(e)}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Save results
        results_file = self.working_dir / 'md_analysis_results.csv'
        df.to_csv(results_file, index=False)
        self.logger.info(f"Results saved to {results_file}")

        # Create Arrhenius plots if we have enough data points
        if len(df) > 1:
            self.plot_arrhenius(df, "diffusion")
            self.plot_arrhenius(df, "conductivity")

            # Calculate activation energy
            slope, _, _, _, _ = stats.linregress(
                df['T_inverse'], df['log10_D'])
            # Ea = -slope * 1000 * np.log(10) * kB * 6.242e18  # Convert to eV
            Ea = -slope * 1000 * np.log(10) * kB
            self.logger.info(f"Activation Energy: {Ea:.2f} eV")

        return df

    def plot_msd_components(self, time_array, msd_data, temperature, filename):
        """Plot MSD components (x, y, z)."""
        fig, ax = plt.subplots()
        for component in ['x', 'y', 'z']:
            ax.plot(time_array, msd_data[component], label=f'MSD_{component}')

        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Å²)')
        ax.set_title(f'{self.structure_name} MSD Components at {temperature}K')
        ax.grid(True)
        ax.legend()

        plt.savefig(
            self.plot_dir / filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def plot_average_msd(
            self,
            time_array,
            avg_msd,
            D_cm2_s,
            temperature,
            filename):
        """Plot average MSD with fit line."""
        fig, ax = plt.subplots()

        # Plot MSD data
        ax.plot(time_array, avg_msd,
                label=f'MSD (D = {D_cm2_s:.2e} cm²/s)')

        # Plot fit line
        slope, intercept = np.polyfit(time_array, avg_msd, 1)
        ax.plot(time_array, slope * time_array + intercept,
                '--', label='Linear fit')

        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Å²)')
        ax.set_title(f'{self.structure_name} Average MSD at {temperature}K')
        ax.grid(True)
        ax.legend()

        plt.savefig(
            self.plot_dir / filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def plot_arrhenius(self, df: pd.DataFrame, plot_type: str):
        """Create Arrhenius plot for either diffusion or conductivity."""
        fig, ax1 = plt.subplots()

        # Setup data based on plot type
        if plot_type == 'diffusion':
            y_data = df['log10_D']
            y_label = 'Log[D (cm² s⁻¹)]'
            title = f'{self.structure_name} Diffusion Arrhenius Plot'
        else:
            y_data = df['log10_conductivity']
            y_label = 'Log[σ (S cm⁻¹)]'
            title = f'{self.structure_name} Conductivity Arrhenius Plot'

        # Calculate activation energy
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['T_inverse'], y_data)
        kb_ev = 8.617333262145e-5  # Boltzmann constant in eV/K
        Ea = -slope * np.log(10) * kb_ev * 1000  # Convert to eV

        # Create main plot
        ax1.scatter(df['T_inverse'], y_data, color='black', s=100)
        ax1.plot(df['T_inverse'], slope * df['T_inverse'] + intercept,
                 '--', color='black', linewidth=2)

        ax1.set_xlabel('1000/T (K⁻¹)')
        ax1.set_ylabel(y_label)

        # Add top x-axis with temperature
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        temp_ticks = [1000 / x for x in ax1.get_xticks()]
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels([f'{int(t)}' for t in temp_ticks])
        ax2.set_xlabel('Temperature (K)')

        # Add activation energy text
        text = f'Ea = {Ea:.2f} eV\nR² = {r_value**2:.3f}'
        ax1.text(0.05, 0.95, text, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.title(title)
        plt.grid(True)

        # Save plot
        plt.savefig(
            self.plot_dir / f'arrhenius_{plot_type}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def export_summary(self, df: pd.DataFrame) -> str:
        """Export analysis summary to text file."""
        summary_file = self.working_dir / 'analysis_summary.txt'

        with open(summary_file, 'w') as f:
            f.write(f"MD Analysis Summary for {self.structure_name}\n")
            f.write("-" * 50 + "\n\n")

            f.write("Analysis Parameters:\n")
            f.write(f"Time step: {self.time_step} fs\n")
            f.write(f"Window size: {self.window_size} steps\n")
            f.write(f"Shift time: {self.shift_time} steps\n")
            f.write(f"Target atom: {self.target_atom}\n\n")

            f.write("Temperature Range Analysis:\n")
            f.write(
                f"Temperature range: {df['temperature'].min()}-{df['temperature'].max()} K\n")
            f.write(f"Number of temperature points: {len(df)}\n\n")

            f.write("Diffusion Analysis:\n")
            f.write(
                f"Average D: {df['D_cm2_s'].mean():.2e} ± {df['D_std'].mean():.2e} cm²/s\n")

            if len(df) > 1:
                slope, _, _, _, _ = stats.linregress(
                    df['T_inverse'], df['log10_D'])
                # Ea = -slope * 1000 * np.log(10) * kB * 6.242e18
                Ea = -slope * 1000 * np.log(10) * kB
                f.write(f"Activation Energy: {Ea:.2f} eV\n\n")

            f.write("Conductivity Analysis:\n")
            f.write(f"Average σ: {df['conductivity'].mean():.2e} S/cm\n")

            # Add detailed results for each temperature
            f.write("\nDetailed Results:\n")
            f.write("-" * 50 + "\n")
            for _, row in df.iterrows():
                f.write(f"\nTemperature: {row['temperature']} K\n")
                f.write(f"MSD: {row['msd_A2ps']:.4f} Å²/ps\n")
                f.write(f"D: {row['D_cm2_s']:.2e} cm²/s\n")
                f.write(f"σ: {row['conductivity']:.2e} S/cm\n")

        return str(summary_file)


def run_analysis(analyzer: MDAnalysisSystem, temperatures: list) -> None:
    """Run diffusion analysis and save results."""
    results_df = analyzer.analyze_temperature_range(temperatures)
    if results_df.empty:
        print("\nNo results to analyze")
        return

    # Save analysis summary
    summary_file = analyzer.export_summary(results_df)
    print(f"\nAnalysis summary saved to: {summary_file}")

    # Print and save results
    print_results(results_df)
    if len(results_df) > 1:
        save_activation_energy(results_df, analyzer.structure_name)


def print_results(results_df: pd.DataFrame) -> None:
    """Print diffusion analysis results."""
    print("\nProton Diffusion Analysis Results:")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"\nTemperature: {row['temperature']} K")
        print(f"H+ Diffusion coefficient: {row['D_cm2_s']:.2e} cm²/s")
        print(f"Proton conductivity: {row['conductivity']:.2e} S/cm")


def save_activation_energy(
        results_df: pd.DataFrame,
        structure_name: str) -> None:
    """Calculate and save activation energy results."""
    slope, _, r_value, _, _ = stats.linregress(
        results_df['T_inverse'],
        results_df['log10_D']
    )
    Ea = -slope * 1000 * np.log(10) * kB

    results_dir = Path("logs/md_analysis")
    results_dir.mkdir(exist_ok=True)

    diffusion_results = {
        'structure': structure_name,
        'temperatures': results_df['temperature'].tolist(),
        'diffusion_coefficients': results_df['D_cm2_s'].tolist(),
        'conductivities': results_df['conductivity'].tolist(),
        'activation_energy_eV': float(Ea),
        'R_squared': float(r_value**2)
    }

    with open(results_dir / 'proton_diffusion_results.json', 'w') as f:
        json.dump(diffusion_results, f, indent=4)


def main():
    """Analyze MD simulation results for proton diffusion."""
    # Setup analysis parameters
    analysis_params = {
        'config': {
            'structures_dir': get_project_paths()['structures_dir'],
            'file_path': get_project_paths()['file_path'],
            'cutoff': 4.0,
            'batch_size': 16,
            'split_ratio': [0.7, 0.1, 0.2],
            'random_state': 42
        },
        'structure_name': "Ba8Zr8O24_H1",
        'target_atom': 'H',
        'time_step': 1.0,
        'shift_time': 50,
        'window_size': 100
    }

    try:
        # Initialize analyzer and run analysis
        analyzer = MDAnalysisSystem(**analysis_params)
        analyzer.working_dir = Path("logs/md_trajectories/Ba8Zr8O24_H2")

        temp_dirs = [
            d for d in analyzer.working_dir.glob("T_*K") if d.is_dir()]
        temperatures = [int(d.name.split('_')[1][:-1]) for d in temp_dirs]

        run_analysis(analyzer, temperatures)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
