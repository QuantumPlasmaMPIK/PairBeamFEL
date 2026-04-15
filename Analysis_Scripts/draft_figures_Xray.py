import sys, os, logging, traceback, pathlib
import glob
import numpy as np
import scipy.constants
import scipy.integrate as integrate
import math
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from matplotlib import colors
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial


logging.basicConfig(level=logging.INFO)
logging.info("Python interpreter: " + sys.executable)
ANALYSIS_DIR_NAME=os.path.dirname(__file__)
logging.info("Directory of the currently executed python script: " + ANALYSIS_DIR_NAME)

## Physical constants in SI units
c  = 299792458             # speed of light in m/s
mu0 = scipy.constants.physical_constants["vacuum mag. permeability"][0] # in N/A^2


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
        logging.error("Diagnostics script could not be found, exit.")
        exit()


########################################################################################################################


## Diagnostics units
unit_dictionary = {"dimensions": ["3d_charge_density", "weight", "angle", "position", "time", "velocity", "current", "energy", "integrated energy", "number density", "average charge", "electric field strength", "magnetic field strength", "Poynting vector"],
                   "units": ["e/cm^3", "m^-1", "degree", "um", "ps", "m/s", "A", "MeV", "joule/um", "m^-1", "C", "V/m", "T", "W/m^2"]
                   }


def show_only_left_bottom(ax):
    # Hide top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Ensure left/bottom are visible
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    # Ticks only on left/bottom to match the spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def make_custom_figure(
    col_widths=(1, 1),        # width ratio for [left, right] in row 1
    row_heights=(2, 1, 1, 1),       # height ratios for [row1, row2]
    figsize=(10, 8)
):
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Top-level grid: 2 rows x 2 columns
    gs = GridSpec(
        nrows=4, ncols=2, figure=fig,
        height_ratios=row_heights, width_ratios=col_widths
    )

    # Row 1, Col 1: single subplot
    ax_left_up = fig.add_subplot(gs[0, 0])
    ax_right_up = fig.add_subplot(gs[0, 1])

    # Row 1, Col 2: nested grid with 3 stacked subplots
    gs_middle = GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs[1, :], hspace=5
    )
    ax_mid1 = fig.add_subplot(gs_middle[0, 0])
    ax_mid2 = fig.add_subplot(gs_middle[0, 1])

    pair = gs[2:,:].subgridspec(2, 1, hspace=0.05)

    # Row 2: one subplot spanning both columns
    #ax_bottom_ep = fig.add_subplot(gs[1, :])
    ax_bottom_ep = fig.add_subplot(pair[0])
    # Row 3: one subplot spanning both columns
    #ax_bottom_e = fig.add_subplot(gs[2, :])
    ax_bottom_e = fig.add_subplot(pair[1])

    return fig, {
        "left_up": ax_left_up,
        "right_up": ax_right_up,
        "mid_left": ax_mid1,
        "mid_right": ax_mid2,
        "bottom_ep": ax_bottom_ep,
        "bottom_e": ax_bottom_e,
    }

fig, axs = make_custom_figure(
    col_widths=(1, 1),       # equal widths in row 1
    row_heights=(2.5, 1.5, 0.65, 0.65),      # taller first row by default
    figsize=(11, 16)
)

########################################################################################################################################################################

#ep_Xray_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/Undulator_reference_Xray_ep/simulation_results/'
#e_Xray_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/Undulator_reference_Xray_e/simulation_results/'
#spike_Xray_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/Undulator_Xray_test/simulation_results/'

e_Xray_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_Xray_e/simulation_results/'
e_Xray_longer_bunch_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_Xray_e_longerbunch/simulation_results/'
ep_Xray_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_Xray_ep/simulation_results/'
ep_Xray_longer_bunch_test_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_Xray_ep_longerbunch/simulation_results/'

simulation_results_paths = [ep_Xray_test_path, e_Xray_test_path]

labels = [r"$e^- \, / \,\, e^+$", r"$e^-$"]
colors = ["coral", "royalblue"]


reference_frame_gamma = 1
reference_frame_velocity = 0

figname = "Xray"

####################################   Temporal spike structure at the exit    #####################################################################################################
# for the spike temporal structure, shorter simulation which just arrived to saturation is considered
simulation_results_paths[0] = ep_Xray_longer_bunch_test_path
simulation_results_paths[1] = e_Xray_longer_bunch_test_path

ax = axs["left_up"]
lws = [2.5, 4.]

timesteps = [23200, 23200]
left_x_crop_percentages = [0.485, 0.485]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
right_x_crop_percentages = [0.78, 0.78]
up_y_crop_percentages = [0.30, 0.30]  ## y boundaries should be the same for proper comparison
down_y_crop_percentages = [0.70, 0.70]
norm_value = 1
bunch_x_min = None
bunch_x_max = None
saturation_timestep = int(14152)
for simulation_results_path_index, simulation_results_path in enumerate(simulation_results_paths):
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

        subset = None
        transverse_axis_name = "y"
        if "3D" in simulation_geometry:
            transverse_axis_name = "z"
        field_names = ["Ey", "Bz_m"]
        unit_attributes = ["electric field strength", "magnetic field strength", "3d_charge_density"]

        timestep = timesteps[simulation_results_path_index]   ## timestep where the bunch just fully exit from the undulator
        TargetFieldDiag = S.Field(0, field="Ey", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        TargetDiag_Ez = S.Field(0, field="Ez", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        TargetDiag_By = S.Field(0, field="By_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        TargetDiag_Bz = S.Field(0, field="Bz_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
        Ez = np.array(TargetDiag_Ez.getData())[0]
        By = np.array(TargetDiag_By.getData())[0]
        Bz = np.array(TargetDiag_Bz.getData())[0]
        labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
        labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
        labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
        labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
        poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0


        #print("Field data shape at a specific time --> ", FieldMatrix.shape)
        FieldMatrix = poynting_flux_amplitude   ## assign matrix to the imaging field
        FieldMatrix = np.rot90(FieldMatrix) # proper rotation
        x_axis = TargetFieldDiag.getAxis("x")
        transverse_axis = TargetFieldDiag.getAxis(transverse_axis_name)

        if simulation_results_path_index == 0:
            diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
            ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
            track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
            weights = track_data["w"]
            bunch_x_min = np.min(track_data["x"])
            bunch_x_max = np.max(track_data["x"])

        left_x_crop_percentage = left_x_crop_percentages[simulation_results_path_index]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
        right_x_crop_percentage = right_x_crop_percentages[simulation_results_path_index]
        x_diff = x_axis[-1] - x_axis[0]
        y_diff = transverse_axis[-1] - transverse_axis[0]
        up_y_crop_percentage = up_y_crop_percentages[simulation_results_path_index]
        down_y_crop_percentage = down_y_crop_percentages[simulation_results_path_index]
        x_bound_min = int(FieldMatrix.shape[0] * up_y_crop_percentage)
        x_bound_max = int(FieldMatrix.shape[0] * down_y_crop_percentage)
        y_bound_min = int(FieldMatrix.shape[1] * left_x_crop_percentage)
        y_bound_max = int(FieldMatrix.shape[1] * right_x_crop_percentage)
        x_axis = x_axis[y_bound_min:y_bound_max]
        normalized_x_axis = (x_axis - bunch_x_min) / (bunch_x_max - bunch_x_min)
        transformed_x_axis = (x_axis - np.min(x_axis)) / (2*reference_frame_gamma) * 1e-6 / c * 1e15   ### in terms of pulse duration in femtoseconds
        print("full range duration in comoving frame (fs.) -->", np.max(transformed_x_axis))
        normalized_y = np.sum(FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], axis=0)
        kernel_size = 15   ### Convolve the pulse
        kernel = np.ones(kernel_size) / kernel_size
        normalized_y = np.convolve(normalized_y, kernel, mode='same')
        if simulation_results_path_index == 0:  # ep case
            norm_value = np.max(normalized_y)
            normalized_y /= norm_value
        else:
            normalized_y /= norm_value

        ax.plot(normalized_x_axis , normalized_y,  ## pulse axis is normalized, as it is directly fetched along the bunch
                lw=lws[simulation_results_path_index], c=colors[simulation_results_path_index], label=labels[simulation_results_path_index])
        print((np.mean(FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], axis=0)).shape)

        if simulation_results_path_index == 0:
            transverse_box_shape = [S.namelist.Main.number_of_cells[1]]
            transverse_box_shape = [S.namelist.Main.number_of_cells[1], S.namelist.Main.number_of_cells[2]]
            poynting_flux_y_limits = [int(0.35 * transverse_box_shape[0]), int(0.65 * transverse_box_shape[0])]
            poynting_flux_z_limits = [int(0.35 * transverse_box_shape[1]), int(0.65 * transverse_box_shape[1])]
            TargetFieldDiag = S.Field(1, field="Ey", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Ez = S.Field(1, field="Ez", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_By = S.Field(1, field="By_m", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Bz = S.Field(1, field="Bz_m", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
            Ez = np.array(TargetDiag_Ez.getData())[0]
            By = np.array(TargetDiag_By.getData())[0]
            Bz = np.array(TargetDiag_Bz.getData())[0]
            labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
            labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
            labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
            labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
            poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0
            poynting_flux_amplitude[poynting_flux_amplitude < 0] = 1e-8
            total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr * S.namelist.Main.cell_length[2] * c / wr # in Watt
            print("peak FEL power (log) --> ", np.log10(total_power))

        #ax.axis('off')
    except Exception as error:
        print("Problem in the transverse bunch profile & bunching factor evolution: ", error)

#ax.set_xlim(left=305, right=310)
#ax.legend(fontsize="medium", loc='upper left', frameon=False, markerscale=20)
ax.set_xlabel(r"bunch coord. ($\mathrm{L_{bunch}}$)", fontsize=28)
ax.set_ylabel("intensity [a.u.]", fontsize=28)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xticks([0, 0.5, 1.])
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels(["0","0.5", "1"], fontsize=28)
ax.set_yticklabels(["0", "0.5", "1"], fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")

ax.legend(fontsize=22, loc=[0.02, 0.71], frameon=False, markerscale=30)

ax.set_ylim(bottom=-0.04)
ax.annotate(
    "", xy=(1.0, -0.02), xytext=(-0.02, -0.02),
    arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=20, color="k")  # width & head size
)
ax.text(0.97, 0.01, "z", fontfamily="sans-serif", fontsize=24, color="k", transform=ax.transAxes)

FWHM_left = 0.62
FWHM_right = 0.72
print("pulse duration in lab frame --> ", np.max(transformed_x_axis) * (FWHM_right - FWHM_left))
ax.annotate(
    "", xy=(FWHM_right, 0.5), xytext=(FWHM_left, 0.5),
    arrowprops=dict(arrowstyle="<->", lw=4, mutation_scale=7, color="k")  # width & head size
)

ax.text(0.48, 0.44, r"$\mathbf{\tau_{FWHM} \, \approx\, 530 \, as}$", fontfamily="sans-serif", fontsize=23, color="k", fontweight="bold", transform=ax.transAxes)
ax.text(0.38, 0.97, r"$\mathbf{P_{sat} \, \approx\, 0.25 \, TW}$", fontfamily="sans-serif", fontsize=24, color="k", fontstyle="italic", fontweight="bold", transform=ax.transAxes)

ax.text(0.02, 0.60, r"$\mathrm{I  \approx 25 \, kA}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)
ax.text(0.02, 0.50, r"$\mathrm{\rho  \approx 3.5\!\times\!10^{-3}}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)
ax.text(0.02, 0.40, r"$\mathrm{L_{b}  \approx 2 \, \mu m}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)
ax.text(0.02, 0.30, r"$\mathrm{Q_{b}  \approx 165 \, pC}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)

ax.text(0.05, 1.01, r"a", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


####################################   Temporal spike structure at the exit    #####################################################################################################
# for the spike temporal structure, shorter simulation which just arrived to saturation is considered
simulation_results_paths[0] = ep_Xray_test_path
simulation_results_paths[1] = e_Xray_test_path

ax = axs["right_up"]
lws = [2.5, 4.]

timesteps = [18400, 18400]
left_x_crop_percentages = [0.50, 0.50]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
right_x_crop_percentages = [0.82, 0.82]
up_y_crop_percentages = [0.30, 0.30]  ## y boundaries should be the same for proper comparison
down_y_crop_percentages = [0.70, 0.70]
norm_value = 1
bunch_x_min = None
bunch_x_max = None
saturation_timestep = int(8464)
for simulation_results_path_index, simulation_results_path in enumerate(simulation_results_paths):
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

        subset = None
        transverse_axis_name = "y"
        if "3D" in simulation_geometry:
            transverse_axis_name = "z"
        field_names = ["Ey", "Bz_m"]
        unit_attributes = ["electric field strength", "magnetic field strength", "3d_charge_density"]

        timestep = timesteps[simulation_results_path_index]   ## timestep where the bunch just fully exit from the undulator
        TargetFieldDiag = S.Field(0, field="Ey", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        TargetDiag_Ez = S.Field(0, field="Ez", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        TargetDiag_By = S.Field(0, field="By_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        TargetDiag_Bz = S.Field(0, field="Bz_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
        FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
        Ez = np.array(TargetDiag_Ez.getData())[0]
        By = np.array(TargetDiag_By.getData())[0]
        Bz = np.array(TargetDiag_Bz.getData())[0]
        labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
        labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
        labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
        labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
        poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0

        #print("Field data shape at a specific time --> ", FieldMatrix.shape)
        FieldMatrix = poynting_flux_amplitude   ## assign matrix to the imaging field
        FieldMatrix = np.rot90(FieldMatrix) # proper rotation
        x_axis = TargetFieldDiag.getAxis("x")
        transverse_axis = TargetFieldDiag.getAxis(transverse_axis_name)

        if simulation_results_path_index == 0:
            diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
            ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
            track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
            weights = track_data["w"]
            bunch_x_min = np.min(track_data["x"])
            bunch_x_max = np.max(track_data["x"])

        left_x_crop_percentage = left_x_crop_percentages[simulation_results_path_index]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
        right_x_crop_percentage = right_x_crop_percentages[simulation_results_path_index]
        x_diff = x_axis[-1] - x_axis[0]
        y_diff = transverse_axis[-1] - transverse_axis[0]
        up_y_crop_percentage = up_y_crop_percentages[simulation_results_path_index]
        down_y_crop_percentage = down_y_crop_percentages[simulation_results_path_index]
        x_bound_min = int(FieldMatrix.shape[0] * up_y_crop_percentage)
        x_bound_max = int(FieldMatrix.shape[0] * down_y_crop_percentage)
        y_bound_min = int(FieldMatrix.shape[1] * left_x_crop_percentage)
        y_bound_max = int(FieldMatrix.shape[1] * right_x_crop_percentage)
        x_axis = x_axis[y_bound_min:y_bound_max]
        normalized_x_axis = (x_axis - bunch_x_min) / (bunch_x_max - bunch_x_min)
        transformed_x_axis = (x_axis - np.min(x_axis)) / (2*reference_frame_gamma) * 1e-6 / c * 1e15   ### in terms of pulse duration in femtoseconds
        print("full range duration in comoving frame (fs.) -->", np.max(transformed_x_axis))
        normalized_y = np.sum(FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], axis=0)
        kernel_size = 15   ### Convolve the pulse
        kernel = np.ones(kernel_size) / kernel_size
        normalized_y = np.convolve(normalized_y, kernel, mode='same')
        if simulation_results_path_index == 0:  # ep case
            norm_value = np.max(normalized_y)
            normalized_y /= norm_value
        else:
            normalized_y /= norm_value

        ax.plot(normalized_x_axis, normalized_y,  ## pulse axis is normalized, as it is directly fetched along the bunch
                lw=lws[simulation_results_path_index], c=colors[simulation_results_path_index], label=labels[simulation_results_path_index])
        print((np.mean(FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], axis=0)).shape)

        if simulation_results_path_index == 0:
            transverse_box_shape = [S.namelist.Main.number_of_cells[1]]
            transverse_box_shape = [S.namelist.Main.number_of_cells[1], S.namelist.Main.number_of_cells[2]]
            poynting_flux_y_limits = [int(0.35 * transverse_box_shape[0]), int(0.65 * transverse_box_shape[0])]
            poynting_flux_z_limits = [int(0.35 * transverse_box_shape[1]), int(0.65 * transverse_box_shape[1])]
            TargetFieldDiag = S.Field(1, field="Ey", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Ez = S.Field(1, field="Ez", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_By = S.Field(1, field="By_m", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Bz = S.Field(1, field="Bz_m", timesteps=saturation_timestep, subset = subset, units=unit_dictionary["units"])
            FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
            Ez = np.array(TargetDiag_Ez.getData())[0]
            By = np.array(TargetDiag_By.getData())[0]
            Bz = np.array(TargetDiag_Bz.getData())[0]
            labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
            labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
            labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
            labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
            poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0
            poynting_flux_amplitude[poynting_flux_amplitude < 0] = 1e-8
            total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr * S.namelist.Main.cell_length[2] * c / wr # in Watt
            print("peak FEL power (log) --> ", np.log10(total_power))

        #ax.axis('off')
    except Exception as error:
        print("Problem in the transverse bunch profile & bunching factor evolution: ", error)

#ax.set_xlim(left=305, right=310)
#ax.legend(fontsize="medium", loc='upper left', frameon=False, markerscale=20)
ax.set_xlabel(r"bunch coord. ($\mathrm{L_{bunch}}$)", fontsize=28)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ticks = [0, 0.5, 1.]
from matplotlib.ticker import FixedLocator, NullFormatter
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([0, 0.5, 1.])
ax.set_xticklabels(["0","0.5", "1"], fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")

ax.legend(fontsize=22, loc=[0.62, 0.71], frameon=False, markerscale=30)

ax.set_ylim(bottom=-0.04)
ax.annotate(
    "", xy=(1.0, -0.02), xytext=(-0.02, -0.02),
    arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=20, color="k")  # width & head size
)
ax.text(0.99, 0.01, "z", fontfamily="sans-serif", fontsize=24, color="k", transform=ax.transAxes)


FWHM_left = 0.45
FWHM_right = 0.57
print("pulse duration in lab frame --> ", np.max(transformed_x_axis) * (FWHM_right - FWHM_left))
ax.annotate(
    "", xy=(FWHM_right, 0.5), xytext=(FWHM_left, 0.5),
    arrowprops=dict(arrowstyle="<->", lw=4, mutation_scale=8, color="k")  # width & head size
)

ax.text(0.43, 0.44, r"$\mathbf{\tau_{FWHM} \, \approx\, 345 \, as}$", fontfamily="sans-serif", fontsize=23, color="k", fontweight="bold", transform=ax.transAxes)
ax.text(0.28, 0.97, r"$\mathbf{P_{sat} \, \approx\, 1.85 \, TW}$", fontfamily="sans-serif", fontsize=24, color="k", fontstyle="italic", fontweight="bold", transform=ax.transAxes)

ax.text(0.015, 0.60, r"$\mathrm{I  \approx 100 \, kA}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)
ax.text(0.015, 0.50, r"$\mathrm{\rho  \approx 6\!\times\!10^{-3}}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)
ax.text(0.015, 0.40, r"$\mathrm{L_{b}  \approx 1 \, \mu m}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)
ax.text(0.005, 0.30, r"$\mathrm{Q_{b}  \approx 330 \, pC}$", fontfamily="sans-serif", fontsize=21, color="k", fontstyle="italic", transform=ax.transAxes)

ax.text(0.05, 1.01, r"b", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


######################################       microbunches      ########################################################################################################################

def phases_from_z(z, k_r):
    """Return phases θ = (k_r z) mod 2π for an array of positions z."""
    return np.mod(k_r * z, 2*np.pi)

def bunching_factor(phases, n=1):
    """Compute |(1/N) sum exp(i n theta)| for given phases theta in radians."""
    return np.abs(np.mean(np.exp(1j * n * phases)))


ax = axs["bottom_ep"]


try:
    # read log-file
    simulation_parameters = {} # to keep important simulation parameters to be used in diagnostics, such as wr for proper unit conversion
    with open(simulation_results_paths[0] + "/simulation_parameters.txt") as f:
        for l in f:
            line_data = l.strip().replace(" ", "").split('=')
            simulation_parameters[line_data[0]] = float(line_data[1])
    print(simulation_parameters)
    wr = None # can be manually set here too
    if "wr" in simulation_parameters:
        wr = simulation_parameters["wr"]
    logging.info("Simulation problem frequency: {}".format(wr))
    # create diagnostics object
    S = happi.Open(simulation_results_paths[0], reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
    logging.info("Happi loads simulation results from the path: " + simulation_results_paths[0])
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

    timestep = 7912
    left_shift = 600
    slice_length = 350
    top = 980
    bottom = 945
    diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
    ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
    track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
    weights = track_data["w"]
    x = track_data["x"]
    z = track_data["z"]
    indexes = np.arange(0, len(weights), dtype=int)  # default indexes
    transformed_axis = x - np.min(x)  # start from 0
    normalized_transverse_axis = (z - bottom) / (top - bottom)
    selected_indexes = np.logical_and(np.logical_and(transformed_axis > left_shift, np.logical_and(transformed_axis < (left_shift + slice_length), normalized_transverse_axis <= 1)), normalized_transverse_axis >= 0)
    transformed_axis = transformed_axis[selected_indexes]
    normalized_transverse_axis = normalized_transverse_axis[selected_indexes]
    ax.scatter(transformed_axis, normalized_transverse_axis, s=0.06, c="dimgray", label = r"$e^-$")
    z_position_holder = transformed_axis.flatten()

    ParticleTrackDiag = S.TrackParticles(species="positron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
    track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
    weights = track_data["w"]
    x = track_data["x"]
    z = track_data["z"]
    indexes = np.arange(0, len(weights), dtype=int)  # default indexes
    transformed_axis = x - np.min(x)
    normalized_transverse_axis = (z - bottom) / (top - bottom)
    selected_indexes = np.logical_and(np.logical_and(transformed_axis > left_shift, np.logical_and(transformed_axis < (left_shift + slice_length), normalized_transverse_axis <= 1)), normalized_transverse_axis >= 0)
    transformed_axis = transformed_axis[selected_indexes]
    normalized_transverse_axis = normalized_transverse_axis[selected_indexes]
    ax.scatter(transformed_axis, normalized_transverse_axis, s=0.06, c="lightseagreen", label = r"$e^+$")
    z_position_holder = transformed_axis.flatten()  #np.concatenate([z_position_holder, transformed_axis.flatten()]).flatten()

    # bunching factor calculation
    um  = 1.e-6 * wr / c            # 1 micrometer in normalized units
    radiation_wavelength_in_labframe = simulation_parameters["FEL_radiation_wl"] / um
    radiation_wavelength_in_bunchrestframe = radiation_wavelength_in_labframe * reference_frame_gamma
    print("Radiation wl in lab frame (um) --> ", radiation_wavelength_in_labframe)
    print("Radiation wl in bunch rest frame (um) & frame gamma --> ", radiation_wavelength_in_bunchrestframe, reference_frame_gamma)
    k_r = 2 * np.pi / radiation_wavelength_in_bunchrestframe
    b_n = bunching_factor(phases_from_z(z_position_holder[z_position_holder > 800], k_r))
    print("bunching factor -> ", b_n)

    
except Exception as error:
    print("Problem in the microbunch profile extraction: ", error)

ax.set_xlim(left=left_shift, right=left_shift+slice_length)
#ax.set_ylim(top=1, bottom=0)
#ax.set_xlabel(r"z [a.u.]", fontsize=28)  ## in undulator setup, mostly 'z' is used as a propagation axis, and y is the transverse axis
ax.set_ylabel(r"y [a.u.]", fontsize=28)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

#ax.set_xticks([0, 200, 400])
from matplotlib.ticker import FixedLocator, NullFormatter
ticks = [left_shift, left_shift+slice_length/2, left_shift+slice_length]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([ 0.5, 1])
#ax.set_xticklabels(["0", "200", "400"], fontsize=28)
ax.set_yticklabels(["0.5", "1"], fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")

ax.legend(fontsize=22, loc=[0.03, 0.10], frameon=True, markerscale=35)

ax.text(0.41, 0.80, r"$\mathrm{b_{n} \approx 0.67 }$", fontfamily="sans-serif", color='indigo', fontsize=26, fontweight="bold",  transform=ax.transAxes)

ax.text(0.95, 0.77, r"e", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


######################################       microbunches      ########################################################################################################################
ax = axs["bottom_e"]

try:
    # read log-file
    simulation_parameters = {} # to keep important simulation parameters to be used in diagnostics, such as wr for proper unit conversion
    with open(simulation_results_paths[1] + "/simulation_parameters.txt") as f:
        for l in f:
            line_data = l.strip().replace(" ", "").split('=')
            simulation_parameters[line_data[0]] = float(line_data[1])
    print(simulation_parameters)
    wr = None # can be manually set here too
    if "wr" in simulation_parameters:
        wr = simulation_parameters["wr"]
    logging.info("Simulation problem frequency: {}".format(wr))
    # create diagnostics object
    S = happi.Open(simulation_results_paths[1], reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
    logging.info("Happi loads simulation results from the path: " + simulation_results_paths[1])
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

    timestep = 7912
    slice_length = 350
    left_shift = 600
    top = 980
    bottom = 945
    diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
    ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
    track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
    weights = track_data["w"]
    x = track_data["x"]
    z = track_data["z"]
    indexes = np.arange(0, len(weights), dtype=int)  # default indexes
    transformed_axis = x - np.min(x)
    normalized_transverse_axis = (z - bottom) / (top - bottom)
    selected_indexes = np.logical_and(np.logical_and(transformed_axis > left_shift, np.logical_and(transformed_axis < (left_shift + slice_length), normalized_transverse_axis <= 1)), normalized_transverse_axis >= 0)
    transformed_axis = transformed_axis[selected_indexes]
    normalized_transverse_axis = normalized_transverse_axis[selected_indexes]
    ax.scatter(transformed_axis, normalized_transverse_axis, s=0.07, c="dimgray", label = r"$e^-$")

    # bunching factor calculation
    um  = 1.e-6 * wr / c            # 1 micrometer in normalized units
    radiation_wavelength_in_labframe = simulation_parameters["FEL_radiation_wl"] / um
    radiation_wavelength_in_bunchrestframe = radiation_wavelength_in_labframe * reference_frame_gamma
    print("Radiation wl in lab frame (um) --> ", radiation_wavelength_in_labframe)
    print("Radiation wl in bunch rest frame (um) & frame gamma --> ", radiation_wavelength_in_bunchrestframe, reference_frame_gamma)
    k_r = 2 * np.pi / radiation_wavelength_in_bunchrestframe
    b_n = bunching_factor(phases_from_z(transformed_axis[:], k_r))
    print("bunching factor -> ", b_n)

except Exception as error:
    print("Problem in the microbunch profile extraction: ", error)

ax.set_xlim(left=left_shift, right=left_shift+slice_length)
ax.set_ylim(top=1, bottom=0)
ax.set_xlabel(r"z ($\mathrm{\mu m}$)", fontsize=28)
ax.set_ylabel(r"y [a.u.]", fontsize=28)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xticks([left_shift, left_shift+slice_length/2, left_shift+slice_length])
ax.set_yticks([0.5, 1])
#ax.set_xticklabels(["0", "0.5", "1"], fontsize=28)
ax.set_yticklabels([ "0.5", "1"], fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")

ax.legend(fontsize=22, loc=[0.03, 0.10], frameon=True, markerscale=35)

ax.text(0.41, 0.80, r"$\mathrm{b_{n} \approx 0.18 }$", fontfamily="sans-serif", color='indigo', fontsize=26, fontweight="bold",  transform=ax.transAxes)

ax.text(0.95, 0.77, r"f", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)



##################################          momentum phase space         ####################################################################################

ax = axs["mid_left"]
color = 'k'
timestep = 0
simulation_id = 0

initial_px = None

try:
    # read log-file
    simulation_parameters = {} # to keep important simulation parameters to be used in diagnostics, such as wr for proper unit conversion
    with open(simulation_results_paths[simulation_id] + "/simulation_parameters.txt") as f:
        for l in f:
            line_data = l.strip().replace(" ", "").split('=')
            simulation_parameters[line_data[0]] = float(line_data[1])
    print(simulation_parameters)
    wr = None # can be manually set here too
    if "wr" in simulation_parameters:
        wr = simulation_parameters["wr"]
    logging.info("Simulation problem frequency: {}".format(wr))
    # create diagnostics object
    S = happi.Open(simulation_results_paths[simulation_id], reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
    logging.info("Happi loads simulation results from the path: " + simulation_results_paths[simulation_id])
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


    diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
    ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
    track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
    weights = track_data["w"]
    px = track_data["px"]
    py = track_data["py"]
    pz = track_data["pz"]
    bunch_gamma = np.sqrt(1 + px**2 + py**2 + pz**2)  # gamma factors
    vx = px / bunch_gamma  # longitudinal velocity (boosted frame)
    transformed_px = reference_frame_gamma * (px - bunch_gamma * -1 * reference_frame_velocity) # lab-frame momentum distr.
    px = transformed_px
    indexes = np.arange(0, len(weights), dtype=int)  # default indexes
    if initial_px is None:
        iniinitial_px = np.mean(px)
    #ax.scatter(px - iniinitial_px, py, s=0.06, c=color)
    from scipy.stats import gaussian_kde
    nbins = 50
    x = px - iniinitial_px
    y = py
    k = gaussian_kde([x,y])
    xi, yi = np.mgrid[
       x.min():x.max():nbins*1j,
       y.min():y.max():nbins*1j
    ]
    zi = k(np.vstack([
       xi.flatten(),
       yi.flatten()
    ])).reshape(xi.shape)
    ax.pcolormesh(xi, yi, zi, cmap=plt.cm.Greens, shading='gouraud')
    r = np.array([1.0, 2.0, 3.0])
    levels = zi.max() * np.exp(-0.5 * r**2)
    print(len(levels), zi.max())
    levels = np.sort(levels)  # must be increasing
    ax.contour(xi, yi, zi, levels=3, linewidths=2., colors='k', linestyles="dotted" )


except Exception as error:
    print("Problem in the microbunch profile extraction: ", error)

ax.set_xlim(left=-10, right=10)
ax.set_ylim(top=0.019, bottom=-0.017)
ax.set_xlabel(r"$\mathrm{p_{\parallel} \, (mc)}$", fontsize=28)
ax.set_ylabel(r"$\mathrm{p_{\perp} \, (mc)}$", fontsize=28)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xticks([-5, 0, 5])
ax.set_yticks([-1e-2, 0, 1e-2])
ax.set_xticklabels([r"$-5$", r"$0$", r"$5$"], fontsize=26)
ax.set_yticklabels([r"$-0.01$", r"$0$", r"$0.01$"], fontsize=26)
ax.tick_params(axis="both", which="major", labelsize=24, length=8, width=1.2, direction="out")


ax.text(0.42, 0.94, r"$\mathrm{\epsilon_{n} = 0.3\,\mu m\text{-}rad}$", fontfamily="sans-serif", color='indigo', fontsize=20,  transform=ax.transAxes)
ax.text(0.42, 0.84, r"$\mathrm{\sigma_{\delta} = 0.1\%}$", fontfamily="sans-serif", color='indigo', fontsize=20,  transform=ax.transAxes)
ax.text(0.35, 0.03, r"lab-frame, at $\mathrm{\tau_{0}}$", fontfamily="sans-serif", color='indigo', fontsize=22,  transform=ax.transAxes)

ax.text(0.05, 1.0, r"c", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


##################################          momentum phase space         ####################################################################################

ax = axs["mid_right"]
colors = ["royalblue", "coral"]
timesteps = [6624, 6624]
simulation_ids = [1, 0]
scatter_sizes = [0.05, 0.05]  # to enhance visibility of the difference scale

zoomed_array = []

initial_px = None
for i in range(len(simulation_ids)):
    color = colors[i]
    timestep = timesteps[i]
    simulation_id = simulation_ids[i]

    try:
        # read log-file
        simulation_parameters = {} # to keep important simulation parameters to be used in diagnostics, such as wr for proper unit conversion
        with open(simulation_results_paths[simulation_id] + "/simulation_parameters.txt") as f:
            for l in f:
                line_data = l.strip().replace(" ", "").split('=')
                simulation_parameters[line_data[0]] = float(line_data[1])
        print(simulation_parameters)
        wr = None # can be manually set here too
        if "wr" in simulation_parameters:
            wr = simulation_parameters["wr"]
        logging.info("Simulation problem frequency: {}".format(wr))
        # create diagnostics object
        S = happi.Open(simulation_results_paths[simulation_id], reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
        logging.info("Happi loads simulation results from the path: " + simulation_results_paths[simulation_id])
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


        diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
        ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        px = track_data["px"]
        py = track_data["py"]
        pz = track_data["pz"]
        indexes = np.arange(0, len(weights), dtype=int)  # default indexes
        if initial_px is not None:
            px -= initial_px
        ax.scatter(px, py, s=scatter_sizes[i], c=color, label = labels[simulation_id])
        zoomed_array.append(px[np.logical_and(px > 0.50, np.abs(py) < 0.50)])
    except Exception as error:
        print("Problem in the microbunch profile extraction: ", error)

ax_hist = ax.inset_axes([0.02, 0.33, 0.55, 0.51])
ax_hist.set_zorder(ax.get_zorder() - 1)
alphas= [0.7, 0.6]
i = 0
colors = ["mediumblue", "orangered"]
for KE in zoomed_array:
    print("zoomed p stat. (mean vs. RMS) -> ", np.mean(KE), np.std(KE))
    simulation_id = simulation_ids[i]
    bins = np.linspace(np.mean(KE) - 3.0*np.std(KE), np.mean(KE) + 3.0*np.std(KE), 30) # create bin to set the interval
    graph, edges = np.histogram(KE, bins, density=True) # create histogram
    normalized_graph = graph / np.max(graph)
    mean_KE = np.mean(KE) # Mean KE value
    mean_KE_index = (np.abs(edges - mean_KE)).argmin()
    mean_KE_edge_value = (edges[mean_KE_index] + edges[mean_KE_index+1]) / 2
    ax_hist.bar(edges[:-1], normalized_graph, width=np.diff(edges), edgecolor="black", lw=0.2, align="edge", color=colors[i], alpha=alphas[i])
    i+=1
ax_hist.set_xlabel(r"$\mathrm{p_{\parallel}}$",  fontsize=20)
ax_hist.xaxis.set_label_coords(0.5, -0.05)
#ax_hist.set_ylabel("normalized amplitude",  fontsize=10)
#ax_hist.set_ylim(bottom=0.2)
ax_hist.set_xticks([0.9, 1.0])
#ax_hist.set_yticks([0.5, 1])
ax_hist.set_xticklabels(["0.9", "1.0"], fontsize=20)
ax_hist.tick_params(axis='x', which='both', pad=0.45)  # smaller = closer to axis
#ax_hist.set_yticklabels(["0.5", "1"], fontsize=16)
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
#ax_hist.xaxis.set_major_formatter(NullFormatter())
#ax_hist.xaxis.set_minor_formatter(NullFormatter())
ax_hist.yaxis.set_major_formatter(NullFormatter())
ax_hist.yaxis.set_minor_formatter(NullFormatter())
ax_hist.yaxis.set_major_locator(NullLocator())
ax_hist.yaxis.set_minor_locator(NullLocator())
#ax_hist.tick_params(top=True, labeltop=False, bottom=False, labelbottom=False)
#ax_hist.minorticks_on()
#ax_hist.set_frame_on(False)
for spine in ax_hist.spines.values():
    spine.set_linewidth(1.2)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)
ax_hist.spines["left"].set_visible(False)
ax_hist.spines["bottom"].set_visible(True)

ax.plot([0.75, 0.75, 1.15, 1.15, 0.75], [-0.5, 0.5, 0.5, -0.5, -0.5], c='dimgrey', lw=2.5)
ax.plot([0.75, -0.45 ], [0.5, 2.30], c='dimgrey', linestyle='--', lw=3)
ax.plot([0.75, 0.2 ], [-0.5, -1.35], c='dimgrey', linestyle='--', lw=3)


#ax.set_xlim(left=0, right=400)
#ax.set_ylim(top=1, bottom=0)
ax.set_xlabel(r"$\mathrm{p_{\parallel} \, (mc)}$", fontsize=28)
#ax.set_ylabel(r"$\mathrm{p_{\perp} \, (mc)}$", fontsize=28)
ax.yaxis.set_label_coords(-0.25, 0.5)  # x<0 moves left; y=0.5 centers vertically
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

#ax.set_xticks([100, 200, 300])
#ax.set_yticks([0, 0.5, 1])
#ax.set_xticklabels(["100", "200", "300"], fontsize=28)
#ax.set_yticklabels(["0", "0.5", "1"], fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=22, loc=[0.64, 0.77], frameon=False, markerscale=35, handletextpad=0.03, borderpad=0.05, labelspacing=0.1)

ax.text(0.40, 0.03, r"rest-frame, at $\mathrm{\tau_{sat.}}$", fontfamily="sans-serif", color='indigo', fontsize=22,  transform=ax.transAxes)

ax.text(0.05, 0.99, r"d", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)



###############################################################################################################################################################################
for ax in axs.values():
    show_only_left_bottom(ax)
    
fig.tight_layout()
pathlib.Path(figname + ".png").unlink(missing_ok=True)
plt.savefig(figname + ".png", format="png", pad_inches=0.02, dpi=250)
plt.close(fig)


