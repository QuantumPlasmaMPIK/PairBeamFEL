## Summary of changes

The customized source tree is based on the public Smilei `v5.1` release. The upstream baseline is the official Smilei `v5.1` tag, with tag commit `438d43d8ab727d02dd8f4402c50393343e871b99`.

No source files were added or removed relative to the baseline `src/` tree. The modifications are limited to the files listed below and implement the Lorentz-boosted undulator-field workflow and the particle-based far-field radiation detector used for the pair-beam FEL simulations.

### Main modification categories

* **Prescribed-field and undulator-field extensions**

  The `PrescribedField` infrastructure was extended so that external fields can either use the original Smilei Python-profile mechanism or a new internal C++ implementation of the Lorentz-boosted undulator field. The new implementation reads the undulator and boost parameters from the namelist and avoids repeated run-time Python profile calls inside the performance-critical field-evaluation loop.

* **Radiation-detector and RaDiO-style far-field support**

  New particle-history and detector-storage arrays were added to support far-field radiation accumulation from particle motion. The customized code stores old momenta, recent deposited fields, detector-pixel positions, detector time grids, and normalization factors needed for the radiation detector. These modifications can be considered as diagnostic additions because the accumulation strategy for far-field radiation signals does not change the conceptual PIC particle pusher.

* **Main-loop integration**

  The radiation detector is initialized during simulation setup, updated during the time loop, and dumped at the end of the run. The modified main loop also ensures that prescribed fields and radiation-detector deposition are applied at controlled points in the simulation workflow.

### Modified files

* `src/ElectroMagn/ElectroMagn.h`

  **Changes:** Extends the `PrescribedField` structure with the new parameters `use_profile`, `frame_gamma`, `frame_velocity`, `undulator_start_lab_frame`, `undulator_end_lab_frame`, `undulator_period`, and `B_peak`; adds the virtual interface `applyPrescribedField(Field*, PrescribedField*, Patch*, double)`.

  **Purpose:** Allows prescribed fields to be evaluated either through the original Smilei Python-profile mechanism or through the new internal C++ Lorentz-boosted undulator-field implementation.

  **Note:** Setting `use_profile = True` keeps the default Smilei behavior, whereas `use_profile = False` activates the internal Lorentz-boosted undulator-field implementation. Hence, to be able to run corresponding simulations, it has to be set to `False`.

* `src/ElectroMagn/ElectroMagn.cpp`

  **Changes:** Updates `ElectroMagn::applyPrescribedFields` to dispatch between the original profile-based field evaluation and the new `PrescribedField`-based C++ evaluation.

  **Purpose:** Preserves the default Smilei prescribed-field behavior while enabling the FEL undulator field to be evaluated directly in C++ for improved performance.

* `src/ElectroMagn/ElectroMagnFactory.h`

  **Changes:** Reads the new `PrescribedField` namelist parameters, including `use_profile`, `frame_gamma`, `frame_velocity`, `undulator_start_lab_frame`, `undulator_end_lab_frame`, `undulator_period`, and `B_peak`, and copies them consistently to patch-local electromagnetic objects.

  **Purpose:** Ensures that Lorentz-boosted undulator-field parameters are initialized once from the namelist and propagated correctly across patches.

* `src/ElectroMagn/ElectroMagn1D.h`

  **Changes:** Adds the new `ElectroMagn1D::applyPrescribedField(Field*, PrescribedField*, Patch*, double)` overload required by the modified electromagnetic interface.

  **Purpose:** Keeps the 1D electromagnetic solver compatible with the extended prescribed-field infrastructure.

* `src/ElectroMagn/ElectroMagn1D.cpp`

  **Changes:** Implements the 1D `applyPrescribedField(Field*, PrescribedField*, Patch*, double)` overload while retaining the standard prescribed-field evaluation on the mesh.

  **Purpose:** Provides interface completeness for 1D simulations without changing the intended 1D prescribed-field behavior.

* `src/ElectroMagn/ElectroMagn2D.h`

  **Changes:** Adds the new `ElectroMagn2D::applyPrescribedField(Field*, PrescribedField*, Patch*, double)` overload for 2D Cartesian geometry.

  **Purpose:** Enables the 2D electromagnetic solver to use the internal C++ Lorentz-boosted undulator-field implementation.

* `src/ElectroMagn/ElectroMagn2D.cpp`

  **Changes:** Implements the Lorentz-boosted planar undulator field in 2D, including lab-frame coordinate reconstruction, finite undulator boundaries, fringe-field treatment, and boosted-frame updates of the relevant electromagnetic components.

  **Purpose:** Provides the 2D FEL undulator field directly in C++, avoiding repeated run-time Python profile calls during field evaluation.

* `src/ElectroMagn/ElectroMagn3D.h`

  **Changes:** Adds the new `ElectroMagn3D::applyPrescribedField(Field*, PrescribedField*, Patch*, double)` overload for 3D Cartesian geometry.

  **Purpose:** Enables the 3D electromagnetic solver to use the internal C++ Lorentz-boosted undulator-field implementation.

* `src/ElectroMagn/ElectroMagn3D.cpp`

  **Changes:** Implements the 3D Lorentz-boosted undulator field, including transverse magnetic-field structure, fringe-field treatment, and boosted-frame updates of the electromagnetic field components.

  **Purpose:** Provides the 3D FEL undulator field directly in C++ for the boosted-frame pair-beam FEL simulations.

* `src/ElectroMagn/ElectroMagnAM.h`

  **Changes:** Adds the new `ElectroMagnAM::applyPrescribedField(Field*, PrescribedField*, Patch*, double)` overload for azimuthal-mode geometry.

  **Purpose:** Keeps the azimuthal-mode electromagnetic solver compatible with the extended prescribed-field interface.

* `src/ElectroMagn/ElectroMagnAM.cpp`

  **Changes:** Adds the profile-versus-internal-field dispatch logic for AM geometry and provides the required `PrescribedField` overload while preserving the existing complex-profile evaluation pathway.

  **Purpose:** Maintains compatibility with Smilei AM prescribed fields while supporting the common modified prescribed-field interface.

* `src/Params/Params.h`

  **Changes:** Adds storage for radiation-detector namelist parameters, including detector-pixel positions, detector time arrays, and physical normalization constants.

  **Purpose:** Stores the detector geometry, detector time sampling, and normalization quantities required for particle-based far-field radiation deposition.

* `src/Params/Params.cpp`

  **Changes:** Reads the `RadiationDetector` namelist block and checks the consistency of detector-pixel coordinate arrays.

  **Purpose:** Makes the far-field radiation detector configurable from the Smilei namelist and validates the detector input geometry at startup.

* `src/Python/pyinit.py`

  **Changes:** Adds the Python-side `RadiationDetector` singleton and extends `PrescribedField` with the Lorentz-boosted undulator parameters exposed to the namelist.

  **Purpose:** Exposes the new radiation-detector and undulator-field options through the standard Smilei namelist interface.

* `src/Particles/Particles.h`

  **Changes:** Adds old-momentum storage and per-particle/per-pixel radiation-deposition state arrays; adds helper access to old momentum storage.

  **Purpose:** Stores particle-history information needed to compute particle acceleration and retarded-time far-field radiation contributions.

* `src/Particles/Particles.cpp`

  **Changes:** Allocates, initializes, copies, and updates old momenta and recent deposited electric/magnetic fields.

  **Purpose:** Enables the radiation detector to compute particle acceleration and avoid double-counting recent field deposition during far-field accumulation.

* `src/Particles/ParticleCreator.cpp`

  **Changes:** Adds comments clarifying that detector-pixel bookkeeping is initialized through the species creation and cloning pathway.

  **Purpose:** Documents the initialization pathway for the radiation-detector particle arrays without changing the functional particle-creation logic.

* `src/Species/SpeciesFactory.h`

  **Changes:** Initializes each species particle container with the radiation-detector pixel count and detector start time, and propagates these quantities to cloned or secondary species.

  **Purpose:** Ensures that all primary and cloned species carry the particle-level storage required by the far-field radiation detector.

* `src/Species/Species.cpp`

  **Changes:** Adds comments clarifying that detector-pixel information is attached to the particle container before initialization.

  **Purpose:** Documents the detector-related initialization order without changing the functional species logic.

* `src/Patch/Patch.h`

  **Changes:** Adds patch-level storage of the global simulation-box size.

  **Purpose:** Makes global box dimensions available inside each patch for customized field and diagnostic calculations.

* `src/Patch/Patch.cpp`

  **Changes:** Copies `params.grid_length` into each patch's `global_grid_length`.

  **Purpose:** Allows patch-local routines to access globally consistent simulation-box dimensions.

* `src/Patch/VectorPatch.h`

  **Changes:** Adds the `RadiationDetector` data structure and declares detector initialization, field deposition, and output routines.

  **Purpose:** Provides the main in-memory detector container and interface used to accumulate far-field radiation from particle motion.

* `src/Patch/VectorPatch.cpp`

  **Changes:** Implements the RaDiO-style detector workflow, including detector initialization, particle-based retarded-time field deposition, thread-private detector workspaces, final detector reduction and output, parallel prescribed-field application, and old-momentum saving before particle pushes.

  **Purpose:** Integrates far-field radiation accumulation into the Smilei patch loop while preserving thread-safe deposition and detector output.

* `src/Smilei.cpp`

  **Changes:** Integrates the radiation detector into the main Smilei workflow by computing normalization factors, initializing the detector, depositing radiation fields during the time loop, applying prescribed fields at a controlled point, and dumping detector output at the end of the simulation.

  **Purpose:** Connects the customized undulator-field and radiation-detector components to the global simulation lifecycle.

