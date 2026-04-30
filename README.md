# PairBeamFEL
Simulation and analysis tools for pair-beam free-electron laser (FEL): 3D particle-in-cell (PIC) approach in the Lorentz-boosted frame with integrated near- and far-field radiation diagnostics.

## Repository structure

This repository provides the simulation inputs, customized PIC source components, and post-processing scripts used for the numerical study. It is organized into three main components:

### 1. `Smilei/`

#### 1.1. `src/`
This directory contains the **modified SMILEI source tree** used for the FEL simulations. It corresponds to a customized implementation of the [public PIC code](https://github.com/SmileiPIC/Smilei) in which the standard source has been extended to support the simulation workflow required for pair-beam FEL studies. In particular, the modified version is intended to handle:

- **Lorentz-boosted undulator-field treatment** for efficient boosted-frame FEL modeling,
- **custom bunch initialization** for electron-only and electron-positron beam configurations in bunch rest frame,
- integration of the **[RaDiO](https://www.sciencedirect.com/science/article/pii/S0010465522003538)-based radiation diagnostic workflow**,
- coupling to the near-field / far-field diagnostic strategy used throughout the paper

Only the relevant source subtree is included here, so this folder should be understood as the code-level component of the reproducibility package rather than as a standalone re-distribution of the full upstream [SMILEI]((https://github.com/SmileiPIC/Smilei)) project.

#### 1.2. `MODIFICATIONS.md`
This file briefly lists the summary of changes, so the differences between the customized and baseline (v5.1) Smilei source code.

#### 1.3. `SLURM_run.sh`
Example batch script for running a job with SLURM manager at high-performance cluster (HPC) in Max Planck Institute for Nuclear Physics. This file should not be considered as reference; it is just a demonstration to show how we use SLURM workload manager while running the Smilei executable with multiple tasks & cores efficiently.

### 2. `Analysis_Scripts/`
This directory contains the **Python post-processing and figure-generation scripts**:

- `draft_figures_UV.py`
- `draft_figures_Xray.py`
- `draft_figures_gamma.py`
- `draft_figures_LSC.py`

These scripts are intended for analysis of the simulation outputs corresponding to the [paper figures](https://arxiv.org/html/2603.15407v1) and include FEL-relevant diagnostics such as beam distributions, power evolution, field structure, and radiation observables. The scripts directly use the **SMILEI Python diagnostics interface (`happi`)** and are written for post-processing previously generated simulation data rather than for fully self-contained execution out of the box. In their current form, they also contain user-side path definitions to simulation result directories, which should be adapted manually before reuse.

At the repository level, these analysis scripts correspond to the paper figure generation workflow for **Fig. 2 to Fig. 5**, respectively.

### 3. `Namelist_Files/`
This directory contains the **input SMILEI namelist files** used to run the different simulation scenarios discussed in the paper:

#### UV regime
- `namelist_UV_ellipsoidal_e.py`
- `namelist_UV_ellipsoidal_ep.py`
- `namelist_UV_pancake_e.py`
- `namelist_UV_pancake_ep.py`

These files correspond to the **UV regular and pancake bunch-shape configurations**, including both **electron-only** (`e`) and **pair-beam** (`ep`) realizations.

#### X-ray regime
- `namelist_Xray_e.py`
- `namelist_Xray_ep.py`
- `namelist_Xray_longerbunch_e.py`
- `namelist_Xray_longerbunch_ep.py`

These files correspond to the **X-ray scenarios**, including **two different current configurations**, again for both electron-only and pair-beam cases.

#### Gamma-ray regime
- `namelist_gamma_e.py`
- `namelist_gamma_ep.py`

These files correspond to the **gamma-regime proof-of-principle simulations** for the electron-only and pair-beam configurations.

The namelists are not generic placeholder inputs: they explicitly document a **Lorentz-boosted SASE FEL setup**, including boosted-frame transformation logic, bunch initialization from lab-frame parameters, prescribed-field handling for the undulator, and external far-field radiation detector support.

## Scenario-to-figure mapping

The repository content is organized to match the simulation campaign reported in the paper:

- **UV regime (regular- and pancake-shape bunches)** → corresponding to the UV comparison figure (`Fig. 2`),
- **X-ray regime (high-current bunches)** → corresponding to the X-ray demonstration figure (`Fig. 3`),
- **gamma regime** → corresponding to the multi-slice gamma-ray figure (`Fig. 4`),

## Notes on use

### Simulation code & SMILEI run
The `Smilei/src/` directory contains the **modified version of the SMILEI source** used in this work. Users intending to reproduce or extend the simulations should treat it as a customized FEL-oriented SMILEI source tree as a code replacement. Build, runtime environment, and external dependencies should therefore be configured consistently with the corresponding SMILEI repo used by the authors, by following public [SMILEI installation page](https://smileipic.github.io/Smilei/Use/installation.html). Input namelist files under `Namelist_Files/` can be used to run simulations, and corresponding [SMILEI run page](https://smileipic.github.io/Smilei/Use/run.html) should be followed how to forward those input files to the executable binary for each above-mentioned scenario.

### Post-processing
The analysis scripts are designed for **diagnostics analysis and figure generation** from existing simulation outputs. In particular:

- they rely on the **[SMILEI diagnostics Python interface (Happi)](https://smileipic.github.io/Smilei/Use/post-processing.html)**,
- they may require local adjustment of file paths to simulation result directories,
- they are intended to be used with simple commandline: `python [script_name]`.

## System requirements

### Baseline code

The customized PIC source tree in `Smilei/src/` is based on the public Smilei `v5.1` source code.

- Upstream baseline: [Smilei v5.1](https://github.com/SmileiPIC/Smilei/releases/tag/v5.1)
- Upstream tag commit: [`438d43d8ab727d02dd8f4402c50393343e871b99`](https://github.com/SmileiPIC/Smilei/commit/438d43d8ab727d02dd8f4402c50393343e871b99)
- Local modifications: FEL-oriented extensions for the Lorentz-boosted undulator setup, bunch initialization, and far-field radiation-diagnostic workflow used in this work.

The full upstream Smilei package is not redistributed here. Users should first obtain and build the public Smilei v5.1 release following the official Smilei installation instructions, then replace the upstream `src/` directory by the customized `Smilei/src/` directory provided in this repository.

### Software dependencies

The code follows the Smilei v5.1 build system and therefore requires the standard Smilei dependencies, including a C++11 compiler, MPI, parallel HDF5, and Python 3. For post-processing, the analysis scripts require Python packages used by the Smilei `happi` interface and by the figure scripts, including `numpy`, `h5py`, and `matplotlib`.

For the full and platform-specific dependency list, see the official [Smilei installation documentation]((https://smileipic.github.io/Smilei/Use/installation.html)).

### Hardware requirements

The analysis scripts can be executed on a standard desktop or laptop once the corresponding simulation output files are available.

The full 3D PIC simulations reported in the manuscript are high-performance-computing simulations and are not intended to be reproduced on a normal desktop computer. The exact runtime depends on the machine, MPI/OpenMP configuration, and selected namelist.

## Installation guide

### 1. Build the upstream Smilei v5.1 code

First, install and test the public Smilei v5.1 release using the official Smilei documentation. A typical CPU build on a recent Linux desktop or workstation takes approximately 5–20 minutes after dependencies are installed. Installation of MPI/HDF5 dependencies may take longer and is system-dependent.

Example:

```bash
git clone https://github.com/SmileiPIC/Smilei.git Smilei_v5.1
cd Smilei_v5.1
git checkout v5.1
make -j 4
make happi
```
The command `make happi` installs the Smilei Python post-processing interface so that `import happi` works from Python. Check the corresponding page for [post-process](https://smileipic.github.io/Smilei/Use/post-processing.html), and also related [Happi installation section](https://smileipic.github.io/Smilei/Use/installation.html#installmodule). Additionally, Smilei’s documentation says compilation is done with `make`, and parallel compilation can be applied by providing the number of cores to the `make` command via `-j` parameter. 

### 2. Replace the source tree by the customized PairBeamFEL source

After confirming that the upstream Smilei v5.1 build works, replace the upstream `src/` directory with the customized source tree provided here:

```bash
cp -r /path/to/PairBeamFEL/Smilei/src /path/to/Smilei_v5.1/
cd /path/to/Smilei_v5.1
make clean
make -j 4
make happi
```
The resulting `smilei` executable is the customized executable used with the namelists in this repository.

## Demo

The lightweight demo consists of running the post-processing scripts on existing Smilei output directories. These scripts reproduce the manuscript-style figures from previously generated simulation data. Before running a script, edit the path variables (`simulation_results_paths`) near the beginning of the script so that they point to the local Smilei output directories.

Example commands:

```bash
python Analysis_Scripts/draft_figures_UV.py
python Analysis_Scripts/draft_figures_Xray.py
python Analysis_Scripts/draft_figures_gamma.py
python Analysis_Scripts/draft_figures_LSC.py
```
Expected runtime on a normal desktop or workstation is typically less than a few minutes per script, assuming the required Smilei output files are already available locally. At present, the repository provides the scripts and namelists but not the full Smilei output datasets because of file-size constraints. A reference dataset suitable for testing the analysis pipeline will be provided with the archival link.

- `draft_figures_UV.py` → `Fig. 2`
- `draft_figures_Xray.py` → `Fig. 3`
- `draft_figures_gamma.py` → `Fig. 4`
- `draft_figures_LSC.py` → `Fig. 5`

## Instructions for use

The complete workflow has three stages.

- **Stage 1**: Produce a customized `smilei` executable
  - build and test Smilei v5.1 by [installing its public release](https://smileipic.github.io/Smilei/Use/installation.html) and confirm that the default executable builds correctly (see `Installation Guide` section)
  - replace the upstream Smilei src/ directory by the customized source tree supplied in this repository (see `Installation Guide` section)
- **Stage 2**: Run a simulation namelist
  - launch the Smilei simulation by passing one of the namelist files in `Namelist_Files/` for a given scenario to the `smilei` executable
  - for production simulations, choose the MPI/OpenMP layout appropriate for the available HPC system --> example sbatch job file in `Smilei/` subfolder (`SLURM_run.sh`) is also provided for demonstration which has been used in HPC of the Max Planck Institute for Nuclear Physics via `SLURM Workload Manager`. For your specific system requirements, also check the corresponding [Smilei run page](https://smileipic.github.io/Smilei/Use/run.html).
    Example:
```
      mkdir run_UV_pancake_ep
      cd run_UV_pancake_ep
      cp /path/to/PairBeamFEL/Namelist_Files/namelist_UV_pancake_ep.py .
      mpirun -n 64 /path/to/Smilei_v5.1/smilei namelist_UV_pancake_ep.py
```
- **Stage 3**: Analyze the output and generate figures
  - edit the path variables in the corresponding analysis script so that they point to the correct output directories (see `Demo` section)
  - execute the python script (see `Demo` section)



## Attribution
**PairBeamFEL** is a scientific project. If you present and/or publish scientific results that benefit from it, we ask you to set a reference to show your support. At present, this can be achieved by simply adding

"Erciyes, Ç., Keitel, C. H., & Tamburini, M., arXiv preprint arXiv:2603.15407 (2026)"

to the references' list of your work. Thank you for your understanding and support!

## Software License
**PairBeamFEL** is licensed under the **GPLv3**.

