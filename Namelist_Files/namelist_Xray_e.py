# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
#
# FEL process with Lorentz-boosted reference frame
#
# Specifications:
# - All simulation parameters are taken as lab-frame, then needed properties are boosted.
# - Electron bunch is initialized via bunch statistical model (shot noise & pre-bunching).
# - Lorentz-transformation is directly considered in the bunch frame, currently no dynamic optimally-chosen frame velocity.
# - Moving window is adaptively enabled.
# - Field profile is customized to be used efficiently within the FEL process.
# - External Radiation Detector block is included.
#
# Validation:
# - Near field radiation power
# - Expected micro-bunching interval and corresponding far-field power spectrum
# - Phase-space trajectories within the beam

#######################################

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

import os, datetime, sys
import math
import numpy as np
import scipy.constants
from scipy.stats import qmc
import matplotlib.pyplot as plt
import logging, traceback
import pathlib
logging.basicConfig(level=logging.INFO)


## Random Number Generator iniital seed
RNG_initial_seed = int(12 * datetime.datetime.today().month + 31 * datetime.datetime.today().day + 24 * datetime.datetime.today().hour)
logging.info("Initial random seed for the Random Number Generators: {:.5e}".format(RNG_initial_seed))



## Beam Species helper class to be used for beam initialization with multiple species
class BeamSpecies:
    def __init__(self, species_name):
        self.species_name = species_name
        self.mass = 0                                           # in electron mass units
        self.v0 = 0                                             # Initial drift velocity in terms of c
        self.gamma = 1                                          # initial lorentz factor (bunch gamma as a default for the electron and positron)
        self.beam_longitudinal_length   = 0                     # longitudinal full-length
        self.beam_transverse_length = 0                         # transverse half-length
        self.bunch_energy_spread        = 0                     # initial rms energy spread / average energy (not in percent)
        self.bunch_normalized_emittance = 0.                    # initial rms emittance, same emittance for both transverse planes (in um_rad)
        self.npart                      = 0                     # number of computational macro-particles to model the bunch
        self.normalized_species_charge  = 0                     # normalized charge (in e)
        self.Q_total  = 0                                       # total charge (in e)
        self.weight  = 0                                        # weight collector
        self.bunch_position_array  = np.array([])               # position collector
        self.bunch_momentum_array  = np.array([])               # momentum collector
        self.longitudinal_spatial_correction = np.nan           # amount of length shrink on lorentz boosted, will be determined later
        def __repr__(self):
            return "Test name:% s charge:% s mass:% s v0:% s" % (self.species_name, self.normalized_species_charge, self.mass, self.v0)

## Statistics collector regarding beam bunching factor
def collect_beam_statistics(k, longitudinal_coordinates, vis=False):
        number_of_buckets = len(np.unique(np.array(longitudinal_coordinates * k / (2 * math.pi) , dtype=int)))
        ponderomotive_phases = k * longitudinal_coordinates
        macroparticle_bucket_indices = np.array(ponderomotive_phases / (2 * math.pi) , dtype=int)
        bucket_indices = np.unique(macroparticle_bucket_indices) # available slice indexes for the whole bunch
        b_n = np.array([]) # complex bunching factor (|b_n|e^{-iQ})
        b_n_modulus = np.array([]) # bunching factor amplitude (|b_n|)
        b_n_modulus_square = np.array([]) # (|b_n|**2)
        for bucket_index in bucket_indices[:-1]:
                collected_phases = ponderomotive_phases[macroparticle_bucket_indices == bucket_index]
                b_n = np.append(b_n, np.mean(np.exp(-1j*collected_phases)))
                b_n_modulus = np.append(b_n_modulus, np.absolute(b_n[-1]))
                b_n_modulus_square = np.append(b_n_modulus_square, b_n_modulus[-1]**2)
        logging.info("bunching_factor (calculated over the whole bunch) -->  {:.5e}".format( np.absolute(np.mean(np.exp(-1j*ponderomotive_phases))) ) )
        logging.info("variance of the bunching factor modulus -->  {:.5e}".format( np.mean(np.square(np.absolute(b_n)) - np.square(np.mean(np.absolute(b_n)))) ) )
        logging.info("mean value of the |B|^2 -->  {:.5e}".format( np.mean(np.square(np.absolute(b_n))) ) )
        if vis:
            histogram_graph = np.histogram(ponderomotive_phases % (2*math.pi), np.linspace(0, 2*math.pi, 50)) # histogram of the resulted ponderomative phases over the whole bunch [0, 2*pi]
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(histogram_graph[1][:-1] + (histogram_graph[1][1] + histogram_graph[1][0])/2, histogram_graph[0])
            ax.set(xlabel=r'$\theta$ (radian)', ylabel='count')
            ax.set_title("Ponderomotive phase distribution over the whole bunch")
            plt.show()
            histogram_graph = np.histogram(np.angle(b_n) + math.pi, np.linspace(np.min(np.angle(b_n) + math.pi), np.max(np.angle(b_n) + math.pi), int(len(np.angle(b_n) + math.pi) / 4) )) # histogram of the resulted bunching factor phases
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(histogram_graph[1][:-1] + (histogram_graph[1][1] + histogram_graph[1][0])/2, histogram_graph[0])
            ax.set(xlabel=r'$\theta$ (radian)', ylabel='count')
            plt.show()
            histogram_graph = np.histogram(b_n_modulus, np.linspace(np.min(b_n_modulus), np.max(b_n_modulus), int(len(b_n_modulus) / 4) )) # histogram of the resulted bunching factor phases
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(histogram_graph[1][:-1] + (histogram_graph[1][1] + histogram_graph[1][0])/2, histogram_graph[0])
            ax.set(xlabel=r'$|b_{n}|$', ylabel='count')
            plt.show()
            histogram_graph = np.histogram(b_n_modulus_square, np.linspace(np.min(b_n_modulus_square), np.max(b_n_modulus_square), int(len(b_n_modulus_square) / 4) )) # histogram of the resulted bunching factors
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(histogram_graph[1][:-1] + (histogram_graph[1][1] + histogram_graph[1][0])/2, histogram_graph[0])
            ax.plot(np.mean(b_n_modulus_square) * np.ones(len(b_n_modulus_square)), np.linspace(np.min(histogram_graph[0]), np.max(histogram_graph[0]), len(b_n_modulus_square)), "--r")
            ax.set(xlabel=r'$|b_{n}|^{2}$', ylabel='count')
            plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


# Physical constants in SI units
c                 = 299792458                       # speed of light in m/s
electron_mass     = 9.10938356e-31                  # electron mass in kg. (scipy.constants.m_e)
electron_charge   = 1.60217662e-19                  # or (scipy.constants.e)
eps0              = scipy.constants.epsilon_0       # Vacuum permittivity, F/m
mu0 = scipy.constants.physical_constants["vacuum mag. permeability"][0]    # in N/A^2
hbar              = 1.054571800e-34                 # (h / 2*pi, plain frequency rather than angular frequency) --> Reduced Planck frequency
pC  = 1.e-12/electron_charge                        # 1 picoCoulomb in normalized units
electron_mass_MeV = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]
MeV = 1./electron_mass_MeV                          # 1 MeV in normalized units
#Hz = 4.13566553853599E-15                          # ! Hz in electron-volt [eV]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

## Simulation box & electron-positron beam properties

# 2d - 3d main simulation feature specifications
simulation_3d             = True
simulation_apply_boost    = True
enable_Radiation_Detector = True
apply_moving_window       = False
position_array_size       = int(4) if simulation_3d else int(3)  #  each dimension grid + weights
momentum_z_include        = True if simulation_3d else False
simulation_geometry       = "3Dcartesian" if simulation_3d else "2Dcartesian"
simulation_BC             = [["silver-muller", "silver-muller"], ["PML", "PML"], ["PML", "PML"]] if simulation_3d else [["PML", "PML"], ["PML", "PML"]]
simulation_PML_cells      = [ [0, 0],  [20, 20], [20, 20]] if simulation_3d else [ [30, 30],  [30, 30]]
simulation_species_BC     = [["remove", "remove"], ["remove", "remove"], ["remove", "remove"]] if simulation_3d else [["remove", "remove"], ["remove", "remove"]]
simulation_relativistic_Poisson_initialization   = True if simulation_3d else False    # in 3d, optimization via relativistic Poisson initializer is important
simulation_Poisson_initialization = False if simulation_relativistic_Poisson_initialization else True   # in 2d, proper field decay is more critical than the usage of relativistic Poisson

# Numerical & model-based algorithms
simulation_particle_pusher = "higueracary"                                 # type of pusher (Boris, borisBTIS3, Vay, higueracary)
simulation_maxwell_solver = "M4"                                    # options: "Yee", "Cowan", "Grassi", "Lehe", "Bouchard", "M4"
simulation_radiation_model = "Landau-Lifshitz"                                  # radiation model for charged species ("Landau-Lifshitz", "CLL", "Niel", "Monte-Carlo")

# Key parameters for the simulation
undulator_period = 30000                                             # laser radiation wavelength in um
wr  = 2.* math.pi * c / (undulator_period * 1e-6)                    # reference frequency of the simulation
K = 3                                                                # undulator parameter
B_peak = K / (0.934 * (undulator_period * 1e-4))                     # undulator magnetic field in Tesla
E_peak = B_peak * c                                                  # # in V/m
bunch_energy = 2000                                                  # initial bunch energy in MeV
bunch_gamma = bunch_energy / electron_mass_MeV                       # initial average lorentz factor for the beam
Q_electron_bunch   = -332 * 1 * pC                                   # Total charge of the electron bunch
beam_longitudinal_sigma = 0.5                                       # Beam longitudinal spatial sigma in um.
beam_transverse_sigma = 40.                                        # Beam transverse/radial rms size (cylindrical symmetry) in um.
beam_fraction = 1                                                 # normalized fraction of the beam that will be simulated, default is 1, i.e. full beam (useful in Xray sim.)
if beam_fraction < 1:                                 # keep the density same, so just cut the beam to focus on the simulated part of it
    Q_electron_bunch *= beam_fraction
    beam_longitudinal_sigma *= beam_fraction
bunch_energy_spread = 1e-3                                           # initial rms energy spread / average energy (not in percent)
bunch_normalized_emittance = 0.3                                     # initial beam normalized emittance (beam divergence X transverse beam size), same for both transverse planes (in um_rad) --> valid for electron & positron
prebunching_factor = 0                                            # Pre-bunched value of the initialized beam
positive_charge_abundance_factor = 1                                # factor (default = 1) that will be multiplied with the total electron charge for a positive charge beam
beam_species = ["electron"]                                          # name of the species contained by the beam


# Main simulation box length, resolution, timing and macroparticle-related parameters
# TODO: Numerical sufficiency can be tested by checking the relative longitudinal movement of the electron beam in bunch frame, net movement should be around zero!
species_macroparticle_number = int(12e6)                              # number of macroparticles to be used to define a species within the beam
num_of_undulator_periods = 275                                       # number of periodic arrangements of dipole magnets with alternating polarity
problem_scale_sampling = 32                                          # number of cells in the lowest problem scale
CFL_multiplication_factor = 0.70                                     # CFL multiplication factor
box_longutidunal_beam_factor = 3.2                                     # simulation_box_x / beam_full_length_x
box_transverse_beam_factor = 16                                      # simulation_box_y / beam_half_length_y
transwverse_cell_number = 128 # np.nan                               # number of cells in transverse direction (either hardcoded or automatic calculation)
diagnostics_frequency = 100                                           # Diagnostics I/O frequency
simulation_time_fraction = 1.4                                        # default is 1.0, mult. factor of T_sim.


# Further derived constants in simulation units
um                     = 1.e-6 * wr / c            # 1 micrometer in normalized units
nm                     = 1.e-9 * wr / c            # 1 nanometer in normalized units
um_rad                = um
fs                     = 1.e-15 * wr                # 1 femtosecond in normalized units
E0_ref                 = electron_mass * wr * c / electron_charge   # reference electric field, V/m
B0_ref                 = electron_mass * wr / electron_charge       # reference magnetic field, V.s/m^2 = Tesla
Schwinger_E_field      = 1.32e18 / E0_ref          # Schwinger electric field (V/m)
Schwinger_B_field      = 4.41e9 / B0_ref           # Schwinger magnetic field (Tesla)
N_r = eps0*(wr**2)*electron_mass/(electron_charge**2)    # Plasma critical number density, m-3


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

## FEL-characteristics logging in addition to beam-plasma properties
ku = 2 * math.pi / (undulator_period / 1e4)   # in cm^-1
bunch_density = -Q_electron_bunch * 1e12 / (math.pi * (2 * beam_transverse_sigma) ** 2 * 2 * beam_longitudinal_sigma)   # in cm^-3
gain_parameter = ((mu0 * K**2 * electron_charge**2 * ku * bunch_density) / (4 * bunch_gamma**3 * electron_mass))**(1./3)   # in cm^-1
gain_length = 1/(math.sqrt(3)*gain_parameter)   # in cm
FEL_parameter = gain_parameter / (2 * ku)
plasma_frequency = math.sqrt(bunch_density * math.pow(electron_charge,2) / (electron_mass * eps0) )
chi = bunch_gamma * B_peak / Schwinger_B_field  # electron quantum parameter
logging.info("Bunch_density, FEL parameter, gain parameter, and gain length: {:.5e} - {:.5e} - {:.5e} - {:.5e}: ".format(bunch_density, FEL_parameter, gain_parameter, gain_length))

## Normalized user inputs to the Smilei units
E_peak /= E0_ref
B_peak /= B0_ref
undulator_period *= um
beam_longitudinal_sigma *= um
beam_transverse_sigma *= um
bunch_normalized_emittance *= um_rad

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

### beam initialization & phase-space parameters
BeamSpeciesDictionary = {}                                  # key: species name, value: corresponding BeamSpecies() object
for species_type in beam_species:                           # search on user configuration
    BeamSpeciesDictionary[species_type] = BeamSpecies(species_type)
if "electron" in BeamSpeciesDictionary:
    BeamSpeciesDictionary["electron"].mass = 1
    BeamSpeciesDictionary["electron"].v0 = math.sqrt(1. - 1. / bunch_gamma ** 2)
    BeamSpeciesDictionary["electron"].gamma = bunch_gamma
    BeamSpeciesDictionary["electron"].beam_longitudinal_length   = 2 * beam_longitudinal_sigma
    BeamSpeciesDictionary["electron"].beam_transverse_length = 3 * beam_transverse_sigma
    BeamSpeciesDictionary["electron"].bunch_energy_spread        = bunch_energy_spread
    BeamSpeciesDictionary["electron"].bunch_normalized_emittance = bunch_normalized_emittance
    BeamSpeciesDictionary["electron"].npart                      = species_macroparticle_number
    BeamSpeciesDictionary["electron"].normalized_species_charge  = 1
    BeamSpeciesDictionary["electron"].Q_total = abs(Q_electron_bunch)
if "positron" in BeamSpeciesDictionary:
    BeamSpeciesDictionary["positron"].mass = 1
    BeamSpeciesDictionary["positron"].v0 = math.sqrt(1. - 1. / bunch_gamma ** 2)
    BeamSpeciesDictionary["positron"].gamma = bunch_gamma
    BeamSpeciesDictionary["positron"].beam_longitudinal_length   = 2 * beam_longitudinal_sigma
    BeamSpeciesDictionary["positron"].beam_transverse_length = 3 * beam_transverse_sigma
    BeamSpeciesDictionary["positron"].bunch_energy_spread        = bunch_energy_spread
    BeamSpeciesDictionary["positron"].bunch_normalized_emittance = bunch_normalized_emittance
    BeamSpeciesDictionary["positron"].npart                      = species_macroparticle_number
    BeamSpeciesDictionary["positron"].normalized_species_charge  = 1
    BeamSpeciesDictionary["positron"].Q_total = abs(Q_electron_bunch) * positive_charge_abundance_factor
if "proton" in BeamSpeciesDictionary:
    BeamSpeciesDictionary["proton"].mass = 1836.15
    BeamSpeciesDictionary["proton"].gamma = 10
    BeamSpeciesDictionary["proton"].v0 = math.sqrt(1. - 1. / BeamSpeciesDictionary["proton"].gamma ** 2)
    BeamSpeciesDictionary["proton"].beam_longitudinal_length   = 2 * beam_longitudinal_sigma
    BeamSpeciesDictionary["proton"].beam_transverse_length = 3 * beam_transverse_sigma
    BeamSpeciesDictionary["proton"].bunch_energy_spread        = bunch_energy_spread
    BeamSpeciesDictionary["proton"].bunch_normalized_emittance = 0
    BeamSpeciesDictionary["proton"].npart                      = species_macroparticle_number
    BeamSpeciesDictionary["proton"].normalized_species_charge  = 1
    BeamSpeciesDictionary["proton"].Q_total = abs(Q_electron_bunch) * positive_charge_abundance_factor

## Beam Initialization in BOOSTED FRAME with initially zero bunching factor via mirroring and then applying shot-noise (or optionally a prebunched mode)
frame_velocity = 0         # no Lorentz-boost initially
frame_gamma = 1
beam_longitudinal_average_velocity = 1 - 1 / (2 * bunch_gamma ** 2) * (1 + K ** 2 / 2)      # average beam velocity across the undulator
beam_longitudinal_average_gamma =  math.sqrt(1 / (1 - (beam_longitudinal_average_velocity ** 2)))      # average Lorentz factor across the undulator
if simulation_apply_boost:  # do it once, take electrons as reference to determine boost velocity
    frame_velocity = beam_longitudinal_average_velocity        # w.r.t. lab
    frame_gamma = math.sqrt(1 / (1 - (frame_velocity ** 2)))   # frame gamma factor w.r.t. lab
cold_plasma_skin_depth = np.nan  # to be filled after retrieving beam properties
cold_plasma_debye_length = np.nan  # to be filled after retrieving beam properties
for species_type in BeamSpeciesDictionary:
    species = BeamSpeciesDictionary[species_type]
    logging.info("Statistical beam initialization is starting for the species " + species_type)
    npart = species.npart
    # initialization
    total_number_simulated_electrons = species.Q_total # total number of electrons in the beam, i.e. total charge
    k_u = 2 * math.pi / undulator_period # calculation should be done in the beam size units, undulator period is given in cm
    FEL_radiation_wl = undulator_period * (1 + K ** 2 / 2) / (2 * bunch_gamma ** 2) # in um
    k_l = 2 * math.pi / FEL_radiation_wl
    beam_center_x = species.beam_longitudinal_length/2
    beam_center_y = beam_transverse_sigma/2
    # mirroring per slice (according to a given highest harmonics) to avoid initial bunching
    #sampler = qmc.Halton(d=1, scramble=False)
    position_x = np.array([])
    highest_harmonic_keep = 10 # to avoid electrostatic field peaks at specific intervals
    bucket_coords = np.arange(0, species.beam_longitudinal_length, 2 * math.pi / (k_u + k_l)) # starting longitudinal x-coordinates of buckets
    number_of_buckets = int(len(bucket_coords) - 1)
    N_e = total_number_simulated_electrons / number_of_buckets # number of electrons per bucket
    for bucket_index, bucket_coord in enumerate(bucket_coords[0:-1]):
            sampling_per_bucket = npart / species.beam_longitudinal_length * (bucket_coords[bucket_index + 1] - bucket_coords[bucket_index]) # number of particles in a slice
            sampling_per_mirror = int(sampling_per_bucket / highest_harmonic_keep / 2) # number of particles in a single mirror
            #sample = sampler.random(n=sampling_per_mirror)
            RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
            RNG_initial_seed += 1
            phase_mirror_intervals = np.linspace(0, 2*math.pi, min(int(highest_harmonic_keep * 2 +1), sampling_per_mirror) ) # mirror phases within the current slice from 0 to 2 pi
            if sampling_per_mirror < highest_harmonic_keep * 2 + 1:
                    logging.warning("\n\n\nNumber of mirror particles is not enough to resolve given number of harmonics \n\n\n")
            phase_mirror_length = phase_mirror_intervals[1] - phase_mirror_intervals[0] # phase interval for the mirroring
            phase_sample = phase_mirror_length * (RNG.random(sampling_per_mirror) - 1./2) + phase_mirror_length / 2 # first sub-slice (starting from phase 0), which will be used as a reference for mirrorring
            position_x = np.append(position_x, bucket_coord + phase_sample / (k_u + k_l)) # position coordinates from the first mirror
            for mirror_index, mirror_coord in enumerate(phase_mirror_intervals[1:-1]):
                    phase_sample += phase_mirror_length
                    position_x = np.append(position_x, bucket_coord + phase_sample / (k_u + k_l)) # position coordinates from the other mirrors within the current slice
    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    RNG.shuffle(position_x)
    npart = len(position_x) # need to be updated due to sampling approach in longitudinal direction
    species.npart = npart
    initial_ponderomotive_phases = position_x * (k_u + k_l)
    logging.info("After phase mirroring initialization, bunching_factor over the whole bunch: {:.5e}".format( np.absolute(np.mean(np.exp(-1j*initial_ponderomotive_phases))) ) )
    # shot-noise (initial random fluctuation) & pre-bunching
    ponderomotive_phases = position_x * (k_u + k_l)
    macroparticle_bucket_indices = np.array(ponderomotive_phases / (2 * math.pi) , dtype=int)
    bucket_indices = np.unique(macroparticle_bucket_indices) # available slice indexes for the whole bunch
    #bunching_sampler = qmc.Halton(d=2, scramble=False)
    #sample = bunching_sampler.random(n=number_of_buckets)

    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    b_n = RNG.random(number_of_buckets)
    b_n[b_n == 0] = np.mean(b_n) # to get rid of numerical values
    b_n = np.sqrt(-np.log(b_n) / N_e)
    logging.info("Sampling check btw. < [b_n]**2 > and 1/N_eb --> {:.5e} {:.5e}".format( np.mean(b_n ** 2), 1. / N_e ) )
    psi_n = np.arcsin(2 * b_n)
    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    phi_n = RNG.random(number_of_buckets)
    phi_n = 2 * math.pi * (phi_n - 1./2) + math.pi
    for bucket_index in bucket_indices:
            if prebunching_factor == 0: # shot-noise
                    position_x[macroparticle_bucket_indices == bucket_index] += 2 * b_n[bucket_index] / (k_u + k_l) * np.sin( ponderomotive_phases[macroparticle_bucket_indices == bucket_index] + phi_n[bucket_index])
            else: # pre-bunched
                    position_x[macroparticle_bucket_indices == bucket_index] += 2 * prebunching_factor / (k_u + k_l) * np.sin( ponderomotive_phases[macroparticle_bucket_indices == bucket_index] )
    position_x -= np.min(position_x) # set the beam starting up to zero
    logging.info("Shot noise is applied into the bunch with a prebunching (modulation) factor: {:.5e}".format( prebunching_factor ) )
    collect_beam_statistics(k_u + k_l, position_x, False)
    # transverse coordinates
    #transverse_sampler = qmc.Halton(d=2, scramble=False)
    #sample = transverse_sampler.random(n=npart)
    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    transverse_sample_y = RNG.random(npart)
    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    transverse_sample_z = RNG.random(npart)
    position_y = beam_transverse_sigma * np.sqrt(-2 * np.log(transverse_sample_y)) * np.cos(2*math.pi*transverse_sample_z) + 0 # transverse profile is always Gaussian
    position_z = beam_transverse_sigma * np.sqrt(-2 * np.log(transverse_sample_y)) * np.sin(2*math.pi*transverse_sample_z) + 0
    position_y[np.isnan(position_y) | np.isinf(position_y)] = np.mean(position_y[np.logical_not(np.isnan(position_y) | np.isinf(position_y))])  # quick fix, check later
    position_z[np.isnan(position_z) | np.isinf(position_z)] = np.mean(position_z[np.logical_not(np.isnan(position_z) | np.isinf(position_z))])  # quick fix, check later
    position_y[position_y > species.beam_transverse_length] = np.mean(position_y) # beam truncation
    position_y[position_y < -species.beam_transverse_length] = np.mean(position_y)
    position_z[position_z > species.beam_transverse_length] = np.mean(position_z)
    position_z[position_z < -species.beam_transverse_length] = np.mean(position_z)


    # Position & Momentum Initialization (the bunch is supposed to be at waist, to make it convergent/divergent Transport Matrices can be used)
    # NOTE: usually density distribution is set by changing the weights not the positions, and so mostly via density profile over a function. However, within the boosted-frame approach, we need array-based initialization
    Q_part                             = species.Q_total / npart                                        # charge per macro-particle in the bunch
    species.weight                     = Q_part/((c/wr)**3 * N_r * species.normalized_species_charge)   # NOTE: for 2d, replace **3 by **2 to be consistent with Units section!
    if not simulation_3d: # weight assignment in 2d geometry
        species.weight                 = Q_part/((c/wr)**2 * N_r * species.normalized_species_charge)
    species.bunch_position_array       = np.zeros((position_array_size, npart))                         # positions x, y, (z), weight
    species.bunch_momentum_array       = np.zeros((3, npart))                                           # momenta x, y, z
    species.bunch_position_array[0, :] = position_x
    species.bunch_position_array[1, :] = position_y
    if simulation_3d:
        species.bunch_position_array[2, :] = position_z
        species.bunch_position_array[3, :] = np.multiply(np.ones(npart), species.weight)
    else:
        species.bunch_position_array[2, :] = np.multiply(np.ones(npart), species.weight)

    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    species.bunch_momentum_array[0, :] = RNG.normal(loc=species.mass * species.v0 * species.gamma, scale= species.mass * species.v0 * species.gamma * species.bunch_energy_spread, size=npart)      # p_x --> generate normal distr. with mean of relativistic momentum (in mc) and std of energy spread
    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    species.bunch_momentum_array[1, :] = RNG.normal(loc=0, scale=species.mass * species.bunch_normalized_emittance / beam_transverse_sigma, size=npart)       # p_y -> generate normal distr. with mean of 0 and std of beam divergence (in rad)
    RNG = np.random.default_rng(RNG_initial_seed)  # single sampling is done for the intial mirror, the rest will follow the phase-shifting
    RNG_initial_seed += 1
    species.bunch_momentum_array[2, :] = RNG.normal(loc=0, scale=species.mass * species.bunch_normalized_emittance / beam_transverse_sigma, size=npart)       # p_z --> generate normal distr. with mean of 0 and std of beam divergence (in rad)
    if not momentum_z_include:
        species.bunch_momentum_array[2, :] = 0


    # Vis.


    ## Lorentz transformation of initial position and velocities, plus convert velocities to momentum with effective lorentz factor (otherwise, only use "longitudinal_spatial_correction" below)
    bunch_time_in_lab_frame = np.zeros(npart)                  # initial time is set to zero in lab frame for all particles
    U_x = np.copy(species.bunch_momentum_array[0, :]) / species.mass                     # x-component of the bunch 4-velocity (momentum given in terms of m*c, so we can directly assign with no need of unit conversion)
    bunch_gamma_factors = np.sqrt(1 + U_x**2)                          # Lorentz factor for each macro-particle (transverse emittance components are ignored)
    bunch_velocities = U_x / bunch_gamma_factors                       # spatial velocity component of the bunch macro-particles
    U_t = np.ones(npart) * bunch_gamma_factors                         # bunch U_0 in lab-frame
    transformed_bunch_U_t = frame_gamma * (U_t - frame_velocity * U_x)   # U'_t
    species.bunch_momentum_array[0, :] = species.mass * frame_gamma * (U_x - frame_velocity * U_t)     # U'_x --> longitudinal momentum transformation, no need for transverse component
    transformed_bunch_longitudinal_velocities = (bunch_velocities - frame_velocity) / (1 - bunch_velocities * frame_velocity)
    transformed_bunch_longitudinal_gammas = 1. / np.sqrt(1 - transformed_bunch_longitudinal_velocities**2) # transformed momentum = transformed gamma X transformed velocity
    transformed_bunch_times = frame_gamma * (bunch_time_in_lab_frame - frame_velocity * species.bunch_position_array[0, :])
    species.bunch_position_array[0, :] *= frame_gamma                        # X' --> Lorentz-transformed spatial position
    species.bunch_position_array[0, :] -= (transformed_bunch_times - np.max(transformed_bunch_times)) * transformed_bunch_longitudinal_velocities # bring them back to the same time instance
    species.longitudinal_spatial_correction = 1. / (frame_gamma * (1 - frame_velocity * species.v0)) # not a direct lorentz transformation, includes both space & time transformation, so bunch length in fixed time
    #species.bunch_position_array[0, :] *= species.longitudinal_spatial_correction
    logging.info("Expected longitudinal spatial correction factor in Lorentz boost for the species {}: {:.5e}".format( species_type, species.longitudinal_spatial_correction ) )
    logging.info("Initial bunch gamma factor in boosted frame for the species {}: {:.5e}".format( species_type, np.mean(transformed_bunch_longitudinal_gammas) ) )


    # Vis.


    ### Check plasma frequency and skin depth values for the plasma in bunch frame, and also compare with cell length below to be sure that plasma is resolved well.
    if species_type == "electron":
        plasma_density_in_bunch_frame = bunch_density / species.longitudinal_spatial_correction # in cm^-3
        debye_factor_for_pair_plasma = math.sqrt(2) # electron-positron plasma
        if "proton" in BeamSpeciesDictionary:
            debye_factor_for_pair_plasma = 1  # electron-ion plasma
        cold_plasma_frequency = debye_factor_for_pair_plasma * 5.64 * 1e4 * math.sqrt(plasma_density_in_bunch_frame)  # in rad/s
        cold_plasma_skin_depth = c * 1e2 / cold_plasma_frequency # in cm
        mean_transverse_velocity = np.mean(np.abs( species.bunch_momentum_array[1, :] ))  # in c
        beam_temperature = mean_transverse_velocity ** 2 * math.pi * 1e6 * scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0] / ( 8 * scipy.constants.physical_constants["Boltzmann constant in eV/K"][0] ) # in K
        cold_plasma_debye_length = 743 * math.sqrt(beam_temperature) / math.sqrt(plasma_density_in_bunch_frame)  # in cm
        logging.info("Cold plasma mean transverse thermal velocity & temperature (in K): {:.3e} & {:.3e}".format( mean_transverse_velocity, beam_temperature ) )
        logging.info("Cold plasma debye length & skin depth (in um): {:.3e} & {:.3e}".format( cold_plasma_debye_length * 1e4 , cold_plasma_skin_depth * 1e4) )



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

### Simulation box and timestepping parameters
labframe_undulator_period = undulator_period    # already given in lab frame
labframe_FEL_radiation_wavelength = labframe_undulator_period / (2 * bunch_gamma ** 2) * (1 + K ** 2 / 2)       #  FEL radiation wl. in lab frame
labframe_bunch_size = BeamSpeciesDictionary["electron"].beam_longitudinal_length                # already given in lab frame
labframe_bunch_velocity = BeamSpeciesDictionary["electron"].v0         # already given in lab frame
bunchframe_FEL_radiation_wavelength = 2 * labframe_FEL_radiation_wavelength * bunch_gamma       #   FEL radiation wl. in bunch frame

reference_undulator_period = labframe_undulator_period / frame_gamma
reference_undulator_length = num_of_undulator_periods * reference_undulator_period
reference_undulator_velocity = -1 * frame_velocity
reference_bunch_size = BeamSpeciesDictionary["electron"].beam_longitudinal_length * BeamSpeciesDictionary["electron"].longitudinal_spatial_correction   # electron beam is considered as reference
reference_longitudinal_bunch_velocity = (labframe_bunch_velocity - frame_velocity) / (1 - frame_velocity * labframe_bunch_velocity)   # longitudinal bunch velocity in reference frame
reference_FEL_radiation_wavelength = bunchframe_FEL_radiation_wavelength * (np.sqrt((1 - reference_longitudinal_bunch_velocity) / (1 + reference_longitudinal_bunch_velocity)))  # case for forward emission, via relativistic transverse doppler effect

undulator_entrance_exit_region = 3 * reference_undulator_period
if simulation_apply_boost:
    required_simulation_time = 0.05 * reference_bunch_size / (abs(reference_longitudinal_bunch_velocity) + abs(reference_undulator_velocity)) + (reference_undulator_length + undulator_entrance_exit_region) / abs(reference_undulator_velocity) # total simulation time
else:
    required_simulation_time = 1. * reference_bunch_size / abs(reference_longitudinal_bunch_velocity) + (reference_undulator_length + undulator_entrance_exit_region) / abs(beam_longitudinal_average_velocity) # total simulation time
required_spatial_resolution = np.minimum(reference_undulator_period, reference_FEL_radiation_wavelength) / problem_scale_sampling
required_spatial_resolution_transverse = 16 * required_spatial_resolution
Lx = box_longutidunal_beam_factor * reference_bunch_size                                           # Dimension of the box in x
Ly = box_transverse_beam_factor * BeamSpeciesDictionary["electron"].beam_transverse_length         # Dimension of the box in y
cell_numbers_x = math.pow(2, math.ceil(math.log2( Lx / required_spatial_resolution )))             # adapt number of cells for sake of patch arrangement
cell_numbers_y = transwverse_cell_number if not np.isnan(transwverse_cell_number) else math.pow(2, math.ceil(math.log2( Ly / required_spatial_resolution_transverse )))  # adapt number of cells for sake of patch arrangement
dx = Lx/cell_numbers_x                          # space step
dy = Ly/cell_numbers_y                          # space step
logging.info("Longitudinal and transverse resolutions (in um): {:.3e}, {:.3e}".format( dx / um, dy / um ) )
logging.info("Bunch frame scale resolution check --> plasma debye length, plasma skin depth, undulator period, FEL radiation wl. (in um) : {:.3e}, {:.3e}, {:.3e}, {:.3e}".format( 1e4 * cold_plasma_debye_length, 1e4 * cold_plasma_skin_depth, reference_undulator_period / um, reference_FEL_radiation_wavelength / um ) )
if simulation_3d:
    dt = CFL_multiplication_factor * 1./math.sqrt(1./(dx*dx) + 1./(dy*dy) + 1./(dy*dy))          # timestep (in grid spacing) or (CFL condition --> 1./math.sqrt(1./(dx*dx) + 1./(dy*dy)))
else:
    dt = CFL_multiplication_factor * 1./math.sqrt(1./(dx*dx) + 1./(dy*dy))
Tsim = simulation_time_fraction * required_simulation_time                 # simulation time


## apply bunch shift to centralize it (currently common for all species)
for species_type in BeamSpeciesDictionary:
    species = BeamSpeciesDictionary[species_type]
    species.bunch_position_array[0, :] += Lx / 2. - np.mean(species.bunch_position_array[0, :]) # shift bunch center to the box center
    species.bunch_position_array[1, :] += Ly / 2. - np.mean(species.bunch_position_array[1, :]) # shift bunch center to the box center
    if simulation_3d:    # include z-shift
        species.bunch_position_array[2, :] += Ly / 2. - np.mean(species.bunch_position_array[2, :]) # shift bunch center to the box center


## undulator start-end positions in lab-frame
undulator_start_reference_frame = np.max(BeamSpeciesDictionary["electron"].bunch_position_array[0, :]) + 2 * reference_undulator_period
undulator_start_lab_frame = frame_gamma * undulator_start_reference_frame
undulator_end_lab_frame = undulator_start_lab_frame + num_of_undulator_periods * labframe_undulator_period



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


## Radiation Detector (RaDiO) related parameters
# TODO: detector start time can be saved as scalar and subtracted from actual array as offset-based approach, so array might start from zero. This resolves Python-based precision issue
r = 1e3 * num_of_undulator_periods * labframe_undulator_period     # should be matched with lab-frame properly due to verification purposes
detector_point_x = r + np.mean(BeamSpeciesDictionary["electron"].bunch_position_array[0, :])     # take the box center as reference
detector_point_y = Ly / 2     # detector central-point is on the x-y plane
if simulation_3d:    # detector z-position
    detector_point_z = Ly / 2
    detector_start_time = np.array([detector_point_x - np.max(BeamSpeciesDictionary["electron"].bunch_position_array[0, :]),
                                    detector_point_y - np.min(BeamSpeciesDictionary["electron"].bunch_position_array[1, :]),
                                    detector_point_z - np.min(BeamSpeciesDictionary["electron"].bunch_position_array[2, :])]) # distance vector
else:
    detector_point_z = 0
    detector_start_time = np.array([detector_point_x - np.max(BeamSpeciesDictionary["electron"].bunch_position_array[0, :]),
                                    detector_point_y - np.min(BeamSpeciesDictionary["electron"].bunch_position_array[1, :]),
                                    0]) # distance vector
detector_start_time = np.linalg.norm(detector_start_time)
gamma_long_avg = 1 if simulation_apply_boost else 1. / math.sqrt(1 - beam_longitudinal_average_velocity**2)
detector_end_time = detector_start_time + Tsim / (gamma_long_avg**2)
time_diff = detector_end_time - detector_start_time
detector_start_time -= time_diff * 1e-2 # put some margins at the end of elapsed time of the detector
detector_end_time += time_diff * 1e-1
detector_time_grid_cell_number = 1e6
detector_time_resolution = (detector_end_time - detector_start_time) / detector_time_grid_cell_number
detector_time_array = np.arange( detector_start_time, detector_end_time, detector_time_resolution, dtype="float64" ) # be careful about time precision; bits needed for actual value & time resolution
if not np.all(np.diff(detector_time_array) >= 0):
    sys.exit("Problem occurs in detector time array...\n")
#np.set_printoptions(precision=15)
#print(detector_time_array)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# seeding
datetime = datetime.datetime.now()
random_seed = datetime.microsecond # random seed prevents the backward trailable on the patch number generators



# ----------------------------------------------------------------------------------------
## LOGGING
np.set_printoptions(precision=12)
logging.info("Associated Simulation frequency: {:.5e}".format(wr))
logging.info("Simulation timestep in seconds: {:.5e}".format(dt*1/wr))
logging.info("Total Simulation time in seconds: {:.5e}".format(Tsim*1/wr))
logging.info("Initial beam lorentz factor: {:.3e}".format(bunch_gamma))
logging.info("Boosted frame lorentz factor: {:.3e}".format(frame_gamma))
logging.info("Beam longitudinal average velocity vs. initial drift velocity: {:.6e} & {:.6e}".format(beam_longitudinal_average_velocity, BeamSpeciesDictionary["electron"].v0))
logging.info("Undulator period, beam length and associated radiation wl: {:.3e}, {:.3e}, {:.3e}".format(undulator_period / um, BeamSpeciesDictionary["electron"].beam_longitudinal_length / um, labframe_FEL_radiation_wavelength / um))
logging.info("Associated K-parameter: {:.3e}: ".format(K))
logging.info("Associated undulator peak magnetic-field strength in Tesla: {:.3e}".format(B_peak * B0_ref))
logging.info("Electron quantum parameter for the simulation settings: {:.3e}".format(chi))
logging.info("Detector mid-point x-y-z: {:.5e} - {:.5e} - {:.5e}".format(detector_point_x, detector_point_y, detector_point_z))
logging.info("Detector time resolution, min & max detection times, obtained time resolution: {:.8e} , {:.12e} & {:.12e}, {:.8e}".format(detector_time_resolution,
                                                                                                                                        detector_time_array[0], detector_time_array[-1],
                                                                                                                                        (detector_time_array[-1] - detector_time_array[0])/len(detector_time_array) )
             )


## Parameter Log
with open(os.getcwd() + "/simulation_parameters.txt", 'w') as f:  #(pathlib.Path(__file__).parent.resolve())
    # problem frequency
    f.write("wr = {:.6e}\n".format(wr))
    # undulator period
    f.write("undulator_period = {}\n".format(undulator_period))
    # FEL primary radiation wavelength
    f.write("FEL_radiation_wl = {}\n".format(FEL_radiation_wl))
    # beam lorentz factor
    f.write("bunch gamma = {}\n".format(bunch_gamma))
    # frame velocity
    f.write("reference_frame_velocity = {}\n".format(frame_velocity))
    # frame lorentz factor
    f.write("reference_frame_gamma = {}\n".format(frame_gamma))
    # K-factor
    f.write("K = {}\n".format(K))
    # labframe undulator start position (x-axis)
    f.write("undulator_start_lab_frame = {}\n".format(undulator_start_lab_frame))
    # beam density (in cm^-3)
    f.write("density = {}\n".format(bunch_density))
    # peak undulator magnetic field (in T)
    f.write("Bz = {}".format(B_peak * B0_ref))
    logging.info("Log file is created to be used for postprocessing...")



# ----------------------------------------------------------------------------------------
## Main simulation parameters


'''
PERFORMANCE OPTIMIZATION
Use dynamic scheduling for the OpenMP protocol!
Use small patches (down to 6x6x6 cells) if simulation has small regions with many particles. Use larger patches (typically 100x100 or 25x25x25 cells) otherwise.
Total number of patches should be larger than total number of threads. Have only as many MPI processes as sockets to optimize memory distribution.
Have only as many MPI processes as sockets in order to optimize the memory distribution. Have as many threads per process as cores per socket.
Vectorized operators w.r.t. their scalar versions can enhance the performance optimization, and factor here depends on ppc (lower performances of vectorized operators at low ppc)! --> For huge simulations you should try it!
'''

patch_number_x_direction = int(min(512, cell_numbers_x/32))
patch_number_y_direction = int(min(64, cell_numbers_y/64))


if simulation_3d:
    simulation_cell_length = [dx,dy,dy]
    simulation_grid_length  = [Lx,Ly,Ly]
    simulation_number_of_patches = [patch_number_x_direction, patch_number_y_direction, patch_number_y_direction] # TODO: patch dimension should be properly chosen for the HPC run
else:
    simulation_cell_length = [dx,dy]
    simulation_grid_length  = [Lx,Ly]
    simulation_number_of_patches = [patch_number_x_direction, patch_number_y_direction]


Main(
    geometry = simulation_geometry,
    interpolation_order = 4,

    cell_length = simulation_cell_length,
    grid_length  = simulation_grid_length,
    number_of_patches = simulation_number_of_patches, # TODO: patch dimension should be properly chosen for the HPC run

    #time_fields_frozen = 0., # Tsim, # this parameter should be used to investigate the effect of interparticle fields!
    solve_poisson = simulation_Poisson_initialization,
    poisson_max_iteration = int(2e5),
    solve_relativistic_poisson  = simulation_relativistic_Poisson_initialization,
    relativistic_poisson_max_iteration = int(2e5),

    reference_angular_frequency_SI = wr,
    timestep = dt,
    simulation_time = Tsim,

    EM_boundary_conditions = simulation_BC,
    number_of_pml_cells    = simulation_PML_cells,

    maxwell_solver=simulation_maxwell_solver,
    #custom_oversize = 6,
    use_BTIS3_interpolation = False, # especially used for BTIS3 pusher scheme

    print_every = 100,
    # random_seed = 0
)


# ----------------------------------------------------------------------------------------
## Define a moving window

if beam_longitudinal_average_velocity == 0:
    logging.warning("\n\n\nAverage longitudinal bunch velocity in the reference frame is equal to zero, no need to use moving window!\n\n\n")
if apply_moving_window:
    MovingWindow(
        time_start =Lx / patch_number_x_direction / beam_longitudinal_average_velocity / 8,          # window starts  moving at the start of the simulation
        velocity_x = beam_longitudinal_average_velocity
    )


# ----------------------------------------------------------------------------------------
## Initialization of the constant external magnetic field


# IMPORTANT: External fields are added to the initial conditions of the fields which are otherwise simply the result of the Poisson solver (in absence of external fields), so they're
# just initial conditions. You can use it to setup a laser pulse inside the box at t=0 or to describe an initial magnetic config. just before a magnetic reconnection event for instance.
# If I'd like to evolve the external field via coupling it with charged particles in a self-consistent way, then "external field" should be used, otherwise "prescribed" should be used as a fixed potential.
# For example, If the undulator is a magnetic field imposed to the particles at all times, then you should indeed set it as prescribed.

'''
laser_duration = 2 * laser_peak_time
laser_center = laser_duration / 2
laser_start = 0
LaserGaussian2D(
    box_side         = "xmax",
    a0              = laser_a0,
    omega           = 1.,
    focus           = [0.5*Lx, 0.5*Ly],
    waist           = 4000 * nm, # this should both conserve the particle interaction dynamics at desired intensity and suppress reflection from boundaries
    incidence_angle = 0.,
    polarization_phi = 0.,
    ellipticity     = 0,
    time_envelope  = tgaussian(start=laser_start,
                               duration= laser_duration,
                               fwhm=laser_FWHM,
                               center=laser_center,
                               order=4)
)
'''




def magnetic_field_profile_3d(x,y,z,t):
    x_lab = frame_gamma * (x + frame_velocity * t) # inverse Lorentz transformation
    k_u = 2 * math.pi / undulator_period   # wave-vector
    if x_lab >= undulator_start_lab_frame and x_lab < undulator_end_lab_frame:
        B_z = B_peak * math.cosh(z) * math.sin(2 * math.pi * (x_lab-undulator_start_lab_frame) / undulator_period)
    else:
        if (x_lab < undulator_start_lab_frame):
            relative_x_lab = x_lab - undulator_start_lab_frame
        else:
            relative_x_lab = x_lab - undulator_end_lab_frame
        B_z = B_peak * math.cosh(z) * k_u * relative_x_lab * math.exp(-1 * math.pow((k_u * relative_x_lab), 2) / 2)
    return frame_gamma * B_z
def magnetic_field_profile(x,y,t):
    x_lab = frame_gamma * (x + frame_velocity * t) # inverse Lorentz transformation
    k_u = 2 * math.pi / undulator_period   # wave-vector
    if x_lab >= undulator_start_lab_frame and x_lab < undulator_end_lab_frame:
        B_z = B_peak * math.cosh(0) * math.sin(2 * math.pi * (x_lab-undulator_start_lab_frame) / undulator_period)
    else:
        if (x_lab < undulator_start_lab_frame):
            relative_x_lab = x_lab - undulator_start_lab_frame
        else:
            relative_x_lab = x_lab - undulator_end_lab_frame
        B_z = B_peak * math.cosh(0) * k_u * relative_x_lab * math.exp(-1 * math.pow((k_u * relative_x_lab), 2) / 2)
    return frame_gamma * B_z


PrescribedField(
    field   = 'Bz_m',
    profile = magnetic_field_profile_3d if simulation_3d else magnetic_field_profile,
    #### Prescribed-field related parameters
    use_profile = False, # default is "True", otherwise the given profile won't be taken into account and internal (currently supports Lorentz-boosted FEL) implementation will be used.
    frame_gamma = frame_gamma,
    frame_velocity = frame_velocity,
    undulator_start_lab_frame = undulator_start_lab_frame,
    undulator_end_lab_frame = undulator_end_lab_frame,
    undulator_period = undulator_period,
    B_peak = B_peak,
)


def electric_field_profile_3d(x,y,z,t):
    x_lab = frame_gamma * (x + frame_velocity * t) # inverse Lorentz transformation
    k_u = 2 * math.pi / undulator_period   # wave-vector
    if x_lab >= undulator_start_lab_frame and x_lab < undulator_end_lab_frame:
        B_z = B_peak * math.cosh(z) * math.sin(2 * math.pi * (x_lab-undulator_start_lab_frame) / undulator_period)
    else:
        if (x_lab < undulator_start_lab_frame):
            relative_x_lab = x_lab - undulator_start_lab_frame
        else:
            relative_x_lab = x_lab - undulator_end_lab_frame
        B_z = B_peak * math.cosh(z) * k_u * relative_x_lab * math.exp(-1 * math.pow((k_u * relative_x_lab), 2) / 2)
    return -1 * frame_gamma * frame_velocity * B_z
def electric_field_profile(x,y,t):
    x_lab = frame_gamma * (x + frame_velocity * t) # inverse Lorentz transformation
    k_u = 2 * math.pi / undulator_period   # wave-vector
    if x_lab >= undulator_start_lab_frame and x_lab < undulator_end_lab_frame:
        B_z = B_peak * math.cosh(0) * math.sin(2 * math.pi * (x_lab-undulator_start_lab_frame) / undulator_period)
    else:
        if (x_lab < undulator_start_lab_frame):
            relative_x_lab = x_lab - undulator_start_lab_frame
        else:
            relative_x_lab = x_lab - undulator_end_lab_frame
        B_z = B_peak * math.cosh(0) * k_u * relative_x_lab * math.exp(-1 * math.pow((k_u * relative_x_lab), 2) / 2)
    return -1 * frame_gamma * frame_velocity * B_z

PrescribedField(
    field   = 'Ey',
    profile = electric_field_profile_3d if simulation_3d else electric_field_profile,
    #### Prescribed-field related parameters
    use_profile = False, # default is "True", otherwise the given profile won't be taken into account and internal (currently supports Lorentz-boosted FEL) implementation will be used.
    frame_gamma = frame_gamma,
    frame_velocity = frame_velocity,
    undulator_start_lab_frame = undulator_start_lab_frame,
    undulator_end_lab_frame = undulator_end_lab_frame,
    undulator_period = undulator_period,
    B_peak = B_peak,
)


if simulation_3d:      # include longitudinal undulator magnetic field only when 3d-sim. is run
    def magnetic_field_longitudinal_profile(x,y,z,t):
        x_lab = frame_gamma * (x + frame_velocity * t) # inverse Lorentz transformation
        if x_lab >= undulator_start_lab_frame and x_lab < undulator_end_lab_frame:
            B_z = B_peak  *  math.sin(2 * math.pi * (x_lab-undulator_start_lab_frame) / undulator_period)
            return frame_gamma * B_z
        else:
            return 0


    PrescribedField(
        field   = 'Bx_m',
        profile = magnetic_field_longitudinal_profile,
        #### Prescribed-field related parameters
        use_profile = False, # default is "True", otherwise the given profile won't be taken into account and internal (currently supports Lorentz-boosted FEL) implementation will be used.
        frame_gamma = frame_gamma,
        frame_velocity = frame_velocity,
        undulator_start_lab_frame = undulator_start_lab_frame,
        undulator_end_lab_frame = undulator_end_lab_frame,
        undulator_period = undulator_period,
        B_peak = B_peak,
    )





# ----------------------------------------------------------------------------------------
## Detector parameters for the numerical radiation implementation (RaDiO)

if enable_Radiation_Detector:
    deviation_angle = 0.015 # in rad.
    RadiationDetector(
        #pixel_positions_x = [detector_point_x, r * np.cos(deviation_angle) + Lx/2, r * np.cos(2 * deviation_angle) + Lx/2],
        #pixel_positions_y = [detector_point_y, detector_point_y, detector_point_y],
        #pixel_positions_z = [detector_point_z, detector_point_z + np.sin(deviation_angle) * r, detector_point_z + np.sin(2*deviation_angle) * r],
        pixel_positions_x = [detector_point_x],
        pixel_positions_y = [detector_point_y],
        pixel_positions_z = [detector_point_z],
        detector_time_array = detector_time_array,
    )



# ----------------------------------------------------------------------------------------
## Create the species

Species(
  name = "electron",
  position_initialization = BeamSpeciesDictionary["electron"].bunch_position_array,
  momentum_initialization = BeamSpeciesDictionary["electron"].bunch_momentum_array,
  #momentum_initialization = "maxwell-juettner",
  #mean_velocity          = [beam_longitudinal_average_velocity, 0, 0],
  #temperature             = [0.00001, 0.00001, 0.],
  mass = 1.0,
  charge = -1.0,
  relativistic_field_initialization = simulation_relativistic_Poisson_initialization,
  pusher = simulation_particle_pusher, # "borisBTIS3", # "ponderomotive_borisBTIS3",
  boundary_conditions = simulation_species_BC,
  radiation_model = simulation_radiation_model,
)

if "positron" in BeamSpeciesDictionary:
    Species(
      name = "positron",
      position_initialization = BeamSpeciesDictionary["positron"].bunch_position_array,
      momentum_initialization = BeamSpeciesDictionary["positron"].bunch_momentum_array,
      #momentum_initialization = "maxwell-juettner",
      #mean_velocity          = [beam_longitudinal_average_velocity, 0, 0],
      #temperature             = [0.00001, 0.00001, 0.],
      mass = 1.0,
      charge = 1.0,
      relativistic_field_initialization = simulation_relativistic_Poisson_initialization,
      pusher = simulation_particle_pusher, # "borisBTIS3", # "ponderomotive_borisBTIS3",
      boundary_conditions = simulation_species_BC,
      radiation_model = simulation_radiation_model,
    )
if "proton" in BeamSpeciesDictionary:
    Species(
      name = "proton",
      position_initialization = BeamSpeciesDictionary["proton"].bunch_position_array,
      momentum_initialization = BeamSpeciesDictionary["proton"].bunch_momentum_array,
      #momentum_initialization = "maxwell-juettner",
      #mean_velocity          = [beam_longitudinal_average_velocity, 0, 0],
      #temperature             = [0.00001, 0.00001, 0.],
      mass = 1836.15,
      charge = 1.0,
      relativistic_field_initialization = simulation_relativistic_Poisson_initialization,
      pusher = simulation_particle_pusher, # "borisBTIS3", # "ponderomotive_borisBTIS3",
      boundary_conditions = simulation_species_BC,
    )


# ----------------------------------------------------------------------------------------

# Radiation Reaction model block
if simulation_radiation_model != "none":
    RadiationReaction(
        minimum_chi_continuous = 1e-10, # TODO: adjust threshold parameter
    )


# ----------------------------------------------------------------------------------------
'''
# Current Filtering to avoid unstable ExB drift of the bunch macro-particles
CurrentFilter(
    model              = "binomial",
    passes             = [2, 0],
    #kernelFIR = [0.25, 0.50, 0.25]
)
'''

# ----------------------------------------------------------------------------------------


## Scalar and Field diagnostics
'''
# Scalars
DiagScalar(
    every = int(Tsim / dt / diagnostics_frequency), # int(oscillation_period/dt*1e-3),
    # "vars" argument is omitted to simply include all available scalars
)
'''
# Fields

# for the longitudinal screen (x-z plane) to check overall back and front emission
longitudinal_field_plane_index = int(np.searchsorted( np.linspace(0, Ly, int(Ly/dy + 1), dtype = "float64"), Ly/2, 'left'))
logging.info("FEL longitudinal field diagnostics screen position is assigned to (index, and in um): {:.1e} {:.5e}".format(longitudinal_field_plane_index, longitudinal_field_plane_index * dy / um))
DiagFields(
    every = int(Tsim / dt / (diagnostics_frequency) ),
    #flush_every           = int(5 * Tsim / dt / diagnostics_frequency ),
    fields = ["Ex", "Ey", "Ez", "By_m", "Bz_m", "Jx_electron", "Rho_electron"],
    subgrid = np.s_[:, longitudinal_field_plane_index, :] if simulation_3d else None # 2d screen (reduction is required for 3d geometry)
)
# for the transverse screen (y-z plane, with some thickness) in front of the beam center
transverse_field_plane_front_index = int(np.searchsorted( np.linspace(0, Lx, int(Lx/dx + 1), dtype = "float64"), Lx/2. + 0.80 * reference_bunch_size, 'left'))
logging.info("FEL transverse field diagnostics screen position is assigned to (in um):  {:.5e} ".format(transverse_field_plane_front_index * dx / um))
DiagFields(
    every = int(Tsim / dt / (diagnostics_frequency) ),
    #flush_every           = int(5 * Tsim / dt / diagnostics_frequency ),
    fields = ["Ey", "Ez", "By_m", "Bz_m", "Jx_electron", "Rho_electron"],
    subgrid = np.s_[transverse_field_plane_front_index, :, :] if simulation_3d else np.s_[transverse_field_plane_front_index, :]  # 2d screen (or 1d line) (reduction is required for 3d geometry)
)


# ----------------------------------------------------------------------------------------
## Particle Binning Diagnostics
# TODO: add its corresponding moving window axis specification into the diagnostics axis

# Energy-distribution
'''
DiagParticleBinning(
    deposited_quantity = "weight",
    every = int(Tsim / dt / diagnostics_frequency),
    time_average = 1,
    species = ["electron"],
    axes = [
        ["ekin", 0.95 * (bunch_gamma - 1), 1.05 * (bunch_gamma - 1), 100],
    ]
)
'''

'''
# Weight x chi spatial-distribution
DiagParticleBinning(
    deposited_quantity = "weight_chi",
    every = int(100), # int(oscillation_period/dt*1e-1),
    time_average = 1,
    species = ["electron"],
    axes = [
        ["moving_x", 0., Lx, 40],
        ["y", 0., Ly, 40],
    ]
)

# Chi-distribution
DiagParticleBinning(
    deposited_quantity = "weight",
    every = int(100), # int(oscillation_period/dt*1e-1),
    time_average = 1,
    species = ["electron"],
    axes = [
        ["chi", 1e-3, 1., 100,"logscale"],
    ]
)
'''
# 2D grid of the weight x normalized kinetic energy (gamma factor - 1) ...


# ----------------------------------------------------------------------------------------

### Tracking Diagnostic for the bunch macro-particles for each species

def my_filter_random_select(particles) :
    ntot = particles.id.size              # total number of particles
    prob = 0.1                            # 10% probability for chosen particles
    mask1 = (particles.id==0)
    mask2 = mask1*(np.random.rand(ntot)<prob)
    particles.id += (72057594037927936*mask1).astype('uint64')

    return (mask2)+(particles.id>72057594037927936)

tracking_attributes = ["x", "y", "px", "py", "w"]
if simulation_3d:
    tracking_attributes = ["x", "y", "z", "px", "py", "pz", "w"]
for species_type in BeamSpeciesDictionary:
    DiagTrackParticles(
        species               = species_type,
        every                 = int(Tsim / dt / diagnostics_frequency),
        #flush_every           = int(5 * Tsim / dt / diagnostics_frequency ),
        attributes            = tracking_attributes,
        filter=my_filter_random_select
    )

# ----------------------------------------------------------------------------------------
## Emitted (Instantaneous) Radiation Diagnostics
#DiagRadiationSpectrum(
#    every = 1,
#    species = ["electron"],
#    photon_energy_axis = [photon_energy_max/1.e6,photon_energy_max, 400, 'logscale'],
#    axes = []
#)


# ----------------------------------------------------------------------------------------
# Probe diagnostics is not needed currently


# ----------------------------------------------------------------------------------------
# Load balancing for the performance optimization.

LoadBalancing(
  initial_balance      = False,
  every                = 200,
  cell_load            = 1.,
  frozen_particle_load = 0.1
)


# ----------------------------------------------------------------------------------------
# Vectorization

if simulation_3d:    # Vectorization optimization makes sense only for 3d simulations currently.
    Vectorization(
        mode = "adaptive",
        reconfigure_every = 25,
        #initial_mode = "on"
    )

# OpenMP task parallelization is experimental ("make config=omptasks" & "make config=part_event_tracing_tasks_on" build configs), should not be used for actual simulations
