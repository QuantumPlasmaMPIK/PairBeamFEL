import sys, os, logging, traceback, pathlib
import glob
import numpy as np
import scipy.constants
import scipy.integrate as integrate
import math
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colors
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial


logging.basicConfig(level=logging.INFO)
logging.info("Python interpreter: " + sys.executable)
ANALYSIS_DIR_NAME=os.path.dirname(__file__)
logging.info("Directory of the currently executed python script: " + ANALYSIS_DIR_NAME)

## Physical constants in SI units
c  = 299792458             # speed of light in m/s
mu0 = scipy.constants.physical_constants["vacuum mag. permeability"][0] # in N/A^2

## Import happi module
try:
    import happi
    logging.info("happi module is accessed via direct import from the python script" )
except Exception as e:
    # Get the diagnostics script to import happi module (specific SMILEI binary version should not matter just to import python module...)
    process_info = None
    diagnostics_script_HPC = "/lfs/l8/theo/quantumplasma/bin/Smilei_5.0/Smilei/scripts/Diagnostics.py" # HPC Smilei bin that I cloned from the Git v5.0 (absolute remote path)
    diagnostics_script_mount = "~/Fileserver/quantumplasma/bin/Smilei_5.0/Smilei/scripts/Diagnostics.py" # HPC Smilei bin that I cloned from the Git v5.0 (mount path)
    diagnostics_script_local = "~/workbench/software/Smilei/scripts/Diagnostics.py" # my own local MPIK Ubuntu Smilei bin (either default or customized SMILEI, does not matter)
    diagnostics_scripts = {"local": diagnostics_script_local, "mount": diagnostics_script_mount, "HPC": diagnostics_script_HPC}
    for script_key in diagnostics_scripts.keys():
        try:
            check_file = os.path.expanduser(diagnostics_scripts[script_key])
            exec(compile(open(check_file).read(), check_file, 'exec'))
            logging.info("Diagnostics script path for the postprecess: " + check_file)
            process_info = script_key
            break
        except Exception as e:
            continue
            #logging.error(traceback.format_exc())
    if process_info is None:
        logging.error("Happi diagnostics script could not be found, exit.")
        exit()


## Load simulation results into the happi diagnostics object
simulation_diagnostics_found = False
simulation_results_path = ANALYSIS_DIR_NAME + "/simulation_data"
try:
    # read log-file
    simulation_parameters = {} # to keep important simulation parameters to be used in diagnostics, such as wr for proper unit conversion
    with open(simulation_results_path + "/simulation_parameters.txt") as f:
        for l in f:
            line_data = l.strip().replace(" ", "").split('=')
            simulation_parameters[line_data[0]] = float(line_data[1])
    print(simulation_parameters)
    wr = None # can be manually set here too
    if "wr" in simulation_parameters:
        wr = simulation_parameters["wr"]
    logging.info("Simulation problem frequency: {}".format(wr))
    # create diagnostics object
    S = happi.Open(simulation_results_path, reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
    logging.info("Happi loads simulation results from the path: " + simulation_results_path)
    simulation_diagnostics_found = True
except Exception as e:
    logging.error(traceback.format_exc())
if not simulation_diagnostics_found:
    sys.exit("Simulation diagnostics can not be found in none of the provided options.")


## Diagnostics units
unit_dictionary = {"dimensions": ["3d_charge_density", "weight", "angle", "position", "time", "velocity", "current", "energy", "integrated energy", "number density", "average charge", "electric field strength", "magnetic field strength", "Poynting vector"],
                   "units": ["e/cm^3", "m^-1", "degree", "um", "ps", "m/s", "A", "MeV", "joule/um", "m^-1", "C", "V/m", "T", "W/m^2"]
                   }


## General info for simulation parameters $ available diagnostics
dt = S.namelist.Main.timestep
dx = S.namelist.Main.cell_length[0]
dy = S.namelist.Main.cell_length[1]
simulation_geometry = S.namelist.Main.geometry
simulation_time = S.namelist.Main.simulation_time
logging.info(' Space steps: %f %f'%(dx,dy))
if "reference_frame_gamma" in simulation_parameters and "reference_frame_velocity" in simulation_parameters:
    reference_frame_gamma = simulation_parameters["reference_frame_gamma"]
    reference_frame_velocity = simulation_parameters["reference_frame_velocity"]
else:
    reference_frame_gamma = 1
    reference_frame_velocity = 0
print("Boosted frame gamma & velocity --> ", reference_frame_gamma, reference_frame_velocity)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

### Field profile (half-grid longitudinal plane is considered)
diag_path = os.path.join(ANALYSIS_DIR_NAME, "diagnostics/Fields_2d")
os.makedirs(diag_path, exist_ok=True)
subset = None
transverse_axis_name = "y"
if "3D" in simulation_geometry:
    transverse_axis_name = "z"
field_names = ["Ey", "Bz_m"]
unit_attributes = ["electric field strength", "magnetic field strength"]
for field_id, field_name in enumerate(field_names):
    timesteps = S.Field(0, field=field_name, units=unit_dictionary["units"]).getTimesteps().astype(int)
    #print("Timesteps for the field screen data --> ", field_name, len(timesteps), timesteps)
    longitudinal_plate_diag_steps = timesteps[:] # []
    for timestep in timesteps:
        if timestep in longitudinal_plate_diag_steps:
            try:
                fig = plt.figure()
                figname = field_name + "_" + str(timestep)
                ax = plt.axes()
                TargetFieldDiag = S.Field(0, field=field_name, timesteps=timestep, subset = subset, units=unit_dictionary["units"])
                FieldMatrix = np.array(TargetFieldDiag.getData())[0]
                #print("Field data shape at a specific time --> ", FieldMatrix.shape)
                FieldMatrix = np.rot90(FieldMatrix) # proper rotation
                x_axis = TargetFieldDiag.getAxis("x")
                transverse_axis = TargetFieldDiag.getAxis(transverse_axis_name)
                ax.set_xlabel("x (" + unit_dictionary["units"][unit_dictionary["dimensions"].index("position")] + ")")
                ax.set_ylabel(transverse_axis_name + " (" + unit_dictionary["units"][unit_dictionary["dimensions"].index("position")] + ")")

                im = ax.imshow(
                    FieldMatrix, cmap="hot", extent=[x_axis[0], x_axis[-1], transverse_axis[0], transverse_axis[-1]], aspect="auto",
                    norm=colors.SymLogNorm( np.abs(np.max(FieldMatrix)) / 1e3)
                )
                cb = fig.colorbar(im, ax=ax, orientation="vertical")
                cb.set_label(unit_dictionary["units"][unit_dictionary["dimensions"].index(unit_attributes[field_id])])
                fig.tight_layout()
                pathlib.Path(diag_path + "/" + figname + ".png").unlink(missing_ok=True)
                plt.savefig(diag_path + "/" + figname + ".png", format="png", dpi=100)
                plt.close()
                print(field_name + " is saved for the timestep: ", timestep)
            except Exception as error:
                print("Zeroth step skipped")



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

### Radiation Detector retrieve the fields and do corresponding analysis
class RaDiO:
    def __init__(self):
        self.time_grid_size = 0
        self.time_array = []
        self.pixel_number = 0
        self.pixel_positions_x = []
        self.pixel_positions_y = []
        self.pixel_positions_z = []
        self.E_x = [] # 2-dim (pixel X time_grid-1)
        self.E_y = []
        self.E_z = []
        self.B_x = [] # 2-dim (pixel X time_grid-1)
        self.B_y = []
        self.B_z = []

RaDiO_fields_dict = {"filenames": [],
                     "deposited_fields": {
                         "ranks": [],
                         "date_times": [],
                         "RaDiO_fields": [],
                     }
                     }

try:
    # read files from each simulation run & rank
    for radiation_file in glob.glob(simulation_results_path + "/Radiation_Detector_*.txt"): # iterate over rank-based field files
        with open(radiation_file, 'rb') as f:
            simulation_ranks = []
            simulation_datetimes = []
            simulation_deposited_fields = []
            RaDiO_fields_dict["filenames"].append(radiation_file)

            lines = f.readlines()   # read the text lines
            line_index = 0
            while line_index < len(lines):
                rank = int(lines[line_index].strip())
                simulation_ranks.append(rank)
                date_time = lines[line_index+1].strip()
                simulation_datetimes.append(date_time)
                RaDiO_field = RaDiO()
                RaDiO_field.time_grid_size = int(lines[line_index+2].strip())
                RaDiO_field.time_array = np.array(lines[line_index+3].strip().split() ).astype(np.double)
                RaDiO_field.pixel_number = int(lines[line_index+4].strip())
                line_index += 5
                for p in range(RaDiO_field.pixel_number):
                    RaDiO_field.pixel_positions_x.append(float(lines[line_index].strip()))
                    RaDiO_field.pixel_positions_y.append(float(lines[line_index+1].strip()))
                    RaDiO_field.pixel_positions_z.append(float(lines[line_index+2].strip()))
                    RaDiO_field.E_x.append( np.array(lines[line_index+3].strip().split() ).astype(np.double) )
                    RaDiO_field.E_y.append( np.array(lines[line_index+4].strip().split() ).astype(np.double) )
                    RaDiO_field.E_z.append( np.array(lines[line_index+5].strip().split() ).astype(np.double) )
                    RaDiO_field.B_x.append( np.array(lines[line_index+6].strip().split() ).astype(np.double) )
                    RaDiO_field.B_y.append( np.array(lines[line_index+7].strip().split() ).astype(np.double) )
                    RaDiO_field.B_z.append( np.array(lines[line_index+8].strip().split() ).astype(np.double) )
                    line_index += 9
                simulation_deposited_fields.append(RaDiO_field)
                logging.info("Radiation Detector is read --> file: {} , rank: {}, datetime: {}".format(os.path.basename(radiation_file), rank, date_time))
            RaDiO_fields_dict["deposited_fields"]["ranks"].append(simulation_ranks)
            RaDiO_fields_dict["deposited_fields"]["date_times"].append(simulation_datetimes)
            RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"].append(simulation_deposited_fields)

except Exception as error:
    print(error)
    sys.exit("Problem in the Radiation Detector diagnostics retrieval")

# combines all the deposited fields linearly into the main Radiation Detector
collected_RaDiO_field = RaDiO() # np.array(RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0] # single file & single rank default local test case
if len(RaDiO_fields_dict["filenames"]) > 0:
    # reshape (common sections are time grid and pixels, then field arrays will be reshaped accordingly to be prepared for linear summation)
    collected_RaDiO_field.time_grid_size = RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0].time_grid_size
    collected_RaDiO_field.time_array = np.array(RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0].time_array)
    collected_RaDiO_field.pixel_number = RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0].pixel_number
    collected_RaDiO_field.pixel_positions_x = np.array(RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0].pixel_positions_x)
    collected_RaDiO_field.pixel_positions_y = np.array(RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0].pixel_positions_y)
    collected_RaDiO_field.pixel_positions_z = np.array(RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][0][0].pixel_positions_z)
    collected_RaDiO_field.E_x = np.zeros((collected_RaDiO_field.pixel_number, collected_RaDiO_field.time_grid_size-1), dtype=np.double)
    collected_RaDiO_field.E_y = np.zeros((collected_RaDiO_field.pixel_number, collected_RaDiO_field.time_grid_size-1), dtype=np.double)
    collected_RaDiO_field.E_z = np.zeros((collected_RaDiO_field.pixel_number, collected_RaDiO_field.time_grid_size-1), dtype=np.double)
    collected_RaDiO_field.B_x = np.zeros((collected_RaDiO_field.pixel_number, collected_RaDiO_field.time_grid_size-1), dtype=np.double)
    collected_RaDiO_field.B_y = np.zeros((collected_RaDiO_field.pixel_number, collected_RaDiO_field.time_grid_size-1), dtype=np.double)
    collected_RaDiO_field.B_z = np.zeros((collected_RaDiO_field.pixel_number, collected_RaDiO_field.time_grid_size-1), dtype=np.double)
    for file_index, filename in enumerate(RaDiO_fields_dict["filenames"]): # over files
        for rank_index, rank_number in enumerate(RaDiO_fields_dict["deposited_fields"]["ranks"][file_index]): # over ranks
            temp_RaDiO_field = RaDiO_fields_dict["deposited_fields"]["RaDiO_fields"][file_index][rank_index]
            for pixel_index in range(temp_RaDiO_field.pixel_number): # over pixels
                collected_RaDiO_field.E_x[pixel_index] += temp_RaDiO_field.E_x[pixel_index]
                collected_RaDiO_field.E_y[pixel_index] += temp_RaDiO_field.E_y[pixel_index]
                collected_RaDiO_field.E_z[pixel_index] += temp_RaDiO_field.E_z[pixel_index]
                collected_RaDiO_field.B_x[pixel_index] += temp_RaDiO_field.B_x[pixel_index]
                collected_RaDiO_field.B_y[pixel_index] += temp_RaDiO_field.B_y[pixel_index]
                collected_RaDiO_field.B_z[pixel_index] += temp_RaDiO_field.B_z[pixel_index]
                logging.info("Collected RaDiO fields --> file: {} , rank: {} , pixel # {}".format(os.path.basename(filename), rank_number, pixel_index))

print("RaDiO fields dictionary: \n", RaDiO_fields_dict)
del RaDiO_fields_dict

# temporal evolution & FFT for a given reference pixel index (results are obtained in lab-frame via inverse Lorentz Transformation)
reference_pixel_index = 0
relativistic_doppler_correction_factor = 1
power_per_solid_angle_correction = 1
if reference_frame_gamma > 1: # not a lab-frame
    relativistic_doppler_correction_factor = 2 * reference_frame_gamma
    power_per_solid_angle_correction = 16 * reference_frame_gamma**4
if collected_RaDiO_field.pixel_number > 0 : # when deposited field exist, do analysis:
    fig, axs = plt.subplots(nrows=2, layout="constrained")
    # time evolution of the electric fields
    signal_time = collected_RaDiO_field.time_array[1:] * 1. / wr   # in sec.
    window_start = int(len(signal_time) * 0.0) # 0
    window_end = int(len(signal_time) * 0.99) # len(signal_time)
    T = signal_time[-1] - signal_time[0]  # total signal elapsed time on the detector
    detector_fetched_time = signal_time[window_start:window_end] - signal_time[window_start]
    detector_fetched_field_Ey = math.sqrt(power_per_solid_angle_correction) * collected_RaDiO_field.E_y[reference_pixel_index][window_start:window_end]
    detector_fetched_field_Ex = math.sqrt(power_per_solid_angle_correction) * collected_RaDiO_field.E_x[reference_pixel_index][window_start:window_end]
    axs[0].plot(detector_fetched_time / np.max(detector_fetched_time), detector_fetched_field_Ey / np.max(detector_fetched_field_Ey), label=r"$E_{y}$")
    axs[0].plot(detector_fetched_time / np.max(detector_fetched_time), detector_fetched_field_Ex / np.max(detector_fetched_field_Ey), label=r"$E_{x}$")
    axs[0].legend(fontsize="small", loc='lower right')
    axs[0].set_xlabel(r"t [a.u.]", fontsize=11)
    axs[0].set_ylabel(r"$|\vec{E}|$ [a.u.]", fontsize=11)
    #axs[0].set_yscale("log")
    # Discrete Fourier Transform, power spectral density, Parseval's theorem check
    signal_amplitude = math.sqrt(power_per_solid_angle_correction) * collected_RaDiO_field.E_y[reference_pixel_index][window_start:window_end]   # Electric field magnitude (considering only the dominant component; plane-wave apprx. will bring multiplication factor of 2 for the spectral density)
    T = signal_time[window_end] - signal_time[window_start]
    N = len(signal_amplitude)
    HF_limit = int(N/4)    # for visualization purposes
    fourierTransform = np.fft.fft(signal_amplitude)/N           # Discrete Fourier Transform and its amplitude normalization
    fourierTransform = fourierTransform[range(int(N/2))]        # exclude sampling frequency
    frequencies      = np.arange(int(N/2)) / T
    power_spectral_density = 2 * np.square(np.abs(fourierTransform))   # d2I/dAdw
    # plot power spectrum with the relativistically Doppler corrected frequencies to get lab-frame reference (assume on-axis pixel currently, off-axis will be checked later)
    collected_freqs = np.array(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0])
    main_freq = collected_freqs[np.argmax(power_spectral_density)]
    axs[1].plot(collected_freqs[:], power_spectral_density[0:HF_limit]) # Normalized PSD with HF-truncated
    axs[1].set_xlim(0.7 * main_freq, 7.5*main_freq)
    axs[1].set_ylim(1e9, 1e14)
    axs[1].set_xlabel(r"eV", fontsize=11)
    axs[1].set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=11)
    axs[1].set_xscale("linear")
    axs[1].set_yscale("log")

    fig.tight_layout()
    pathlib.Path(ANALYSIS_DIR_NAME + "/diagnostics//RaDiO_Detector.png").unlink(missing_ok=True)
    plt.savefig("diagnostics/RaDiO_Detector.png", format="png", dpi=400)
    #plt.show(block=True)
    plt.close()
    intensity_temporal_amplitude = np.square( np.sqrt( collected_RaDiO_field.E_x[reference_pixel_index]**2 + collected_RaDiO_field.E_y[reference_pixel_index]**2 + collected_RaDiO_field.E_z[reference_pixel_index]**2 ) )
    power_density = integrate.simpson(intensity_temporal_amplitude, signal_time)  #dI/dA
    # Parseval's theorem, when applied to DFTs, doesn't require integration, but summation: a 2*pi normalization factor pops up by multipliying by dt and df your summations.
    # The Plancherel theorem is a special case of Parseval's theorem (DFT case)
    temporal_intensity = np.square(signal_amplitude)       # considering only the dominant electric field component here, which should not affect the temporal structure.
    fourierTransform = np.fft.fft(signal_amplitude)
    sampling_threshold = N
    fourierTransform = fourierTransform[range(int(sampling_threshold))]        # Exclude sampling frequency
    frequencies      = np.arange(int(sampling_threshold)) / T
    power_spectral_density = np.square(np.abs(fourierTransform))   # d2I/dAdw
    parseval_w = np.sum(power_spectral_density)/N
    parseval_t = np.sum(temporal_intensity)
    logging.info("Parseval (DFT) check --> parseval_t: {} , parseval_w: {} , relative_fraction: {}".format(parseval_t, parseval_w, parseval_t/parseval_w))
    del fig, axs


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
