# PairBeamFEL
Simulation and analysis tools for pair-beam free-electron laser (FEL): 3D particle-in-cell (PIC) approach in the Lorentz-boosted frame with integrated near- and far-field radiation diagnostics.

## Repository structure

This repository provides the simulation inputs, customized PIC source components, and post-processing scripts used for the numerical study. It is organized into three main components:

### 1. `Smilei/src/`
This directory contains the **modified SMILEI source tree** used for the FEL simulations. It corresponds to a customized implementation of the [public PIC code](https://github.com/SmileiPIC/Smilei) in which the standard source has been extended to support the simulation workflow required for pair-beam FEL studies. In particular, the modified version is intended to handle:

- **Lorentz-boosted undulator-field treatment** for efficient boosted-frame FEL modeling,
- **custom bunch initialization** for electron-only and electron-positron beam configurations in bunch rest frame,
- integration of the **[RaDiO](https://www.sciencedirect.com/science/article/pii/S0010465522003538)-based radiation diagnostic workflow**,
- coupling to the near-field / far-field diagnostic strategy used throughout the paper

Only the relevant source subtree is included here, so this folder should be understood as the code-level component of the reproducibility package rather than as a standalone re-distribution of the full upstream [SMILEI]((https://github.com/SmileiPIC/Smilei)) project.

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

## Attribution
**PairBeamFEL** is a scientific project. If you present and/or publish scientific results that benefit from it, we ask you to set a reference to show your support. At present, this can be achieved by simply adding

"Erciyes, Ç., Keitel, C. H., & Tamburini, M., arXiv preprint arXiv:2603.15407 (2026)"

to the references' list of your work. Thank you for your understanding and support!

## Software License
**PairBeamFEL** is licensed under the **GPLv3**.

