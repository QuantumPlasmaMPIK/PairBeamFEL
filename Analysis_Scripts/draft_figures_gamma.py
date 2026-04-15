import sys, os, logging, traceback, pathlib
import glob
import numpy as np
import scipy.constants
import scipy.integrate as integrate
import math
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from matplotlib.colors import SymLogNorm
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

########################################################################################################################################################################
three_slice_ep_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_Gamma_ep/simulation_results/'
three_slice_e_path = '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_Gamma_e/simulation_results/'
simulation_results_paths = [three_slice_ep_path, three_slice_e_path]

labels = [r"$e^- \, / \,\, e^+$", r"$e^-$"]
colors = ["coral", "royalblue"]


reference_frame_gamma = 1
reference_frame_velocity = 0

def make_4row_3col_figure(
    figsize=(14, 14),
    height_ratio=(2.5, 2.5, 2.5, 2),      # first row : second row
    wspace=0.1,
    hspace=0.2,
    sharex=False,
    sharey=False
):
    """
    Create a figure with:
      - Row 1: one axis spanning full width (no columns)
      - Row 2: three equal-width axes
    Row heights are adjustable via height_ratio=(top, bottom).
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(
        nrows=4,
        ncols=3,
        figure=fig,
        height_ratios=[height_ratio[0], height_ratio[1], height_ratio[2], height_ratio[3]],
        width_ratios=[1, 1, 1]
    )
    # spacing
    #gs.update(hspace=hspace)

    # Top axis spans all columns of the first row
    ax_top_1 = fig.add_subplot(gs[0, :])
    ax_top_2 = fig.add_subplot(gs[1, :])
    ax_top_3 = fig.add_subplot(gs[2, :])

    # Three equal subplots in the second row
    sub = gs[3, :].subgridspec(nrows=10, ncols=3, wspace=0.125)
    ax_b1 = fig.add_subplot(sub[0:10, 0])
    ax_b2 = fig.add_subplot(sub[:, 1])
    ax_b3 = fig.add_subplot(sub[:, 2])

    #ax_b1 = fig.add_subplot(gs[2, 0])
    #rightpair = gs[2, 1:].subgridspec(1, 2, wspace=0.10)
    #ax_b2 = fig.add_subplot(rightpair[0, 0])
    #ax_b3 = fig.add_subplot(rightpair[0, 1], sharey=ax_b2)

    # optional: hide duplicate left labels on the shared-right axis
    ax_b3.tick_params(labelleft=False)

    axes = {
        "top1": ax_top_1,
        "top2": ax_top_2,
        "top3": ax_top_3,
        "bottom": (ax_b1, ax_b2, ax_b3)
    }
    return fig, axes


fig, axs = make_4row_3col_figure(figsize=(12.5, 12.5), height_ratio=(1.5, 1.5, 1.5, 2.5))
figname = "Gamma"


####################################   Temporal spike structure for multi spike (different regimes)   #####################################################################################################
ax = axs["top1"]
lws = [2.5, 2.5]
linecolors = ["whitesmoke", "lime"]

timesteps = [50344]
left_x_crop_percentages = [0.20]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
right_x_crop_percentages = [0.85]
up_y_crop_percentages = [0.40]
down_y_crop_percentages = [0.60]
norm_value = 1
line_plots = []
sc = None

simulation_results_path_index = 0
simulation_results_path = simulation_results_paths[0]
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
    print("simulation time at this timestep (in sec.)" , timesteps[0] * dt / wr)
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
    prop_sign = (FieldMatrix * Bz - Ez * By) < 0  # negative propagation which will be filtered later
    labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
    labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
    labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
    labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
    poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0
    poynting_flux_amplitude[prop_sign] = 0


    #print("Field data shape at a specific time --> ", FieldMatrix.shape)
    FieldMatrix = poynting_flux_amplitude   ## assign matrix to the imaging field
    FieldMatrix = np.rot90(FieldMatrix) # proper rotation
    x_axis = TargetFieldDiag.getAxis("x")
    transverse_axis = TargetFieldDiag.getAxis(transverse_axis_name)

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
    transverse_axis = transverse_axis[x_bound_min:x_bound_max]
    x_diff = x_axis[-1] - x_axis[0]
    y_diff = transverse_axis[-1] - transverse_axis[0]
    transverse_axis_normalized = transverse_axis #(transverse_axis - transverse_axis.min()) / (transverse_axis.max() - transverse_axis.min()) * 2 - 1  # normalize for plotting purposes

    im = ax.imshow(
        FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], cmap="inferno", extent=[x_axis[0], x_axis[-1], transverse_axis_normalized[0], transverse_axis_normalized[-1]],
                   aspect="auto", norm=SymLogNorm( np.abs(np.max(FieldMatrix)) / 250)
    )
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    import matplotlib.ticker as mticker
    ticks = [1e21, 1e22, 1e23]
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    cb.set_ticks(ticks)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
    cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
    cb.set_label(r"$\mathrm{W\,/\,m^{2}}$", fontsize=18, labelpad=10)
    
    diagnostics_axes = ["moving_x", "x", "y", "z", "w"]
    species_list = ["electron", "electron_1", "electron_2"]
    for species in species_list:
        ParticleTrackDiag = S.TrackParticles(species=species, timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        indexes = np.arange(0, len(weights), dtype=int)  # default indexes
        bunch_x = track_data["x"][indexes]
        bunch_y = track_data[transverse_axis_name][indexes]
        bunch_y_normalized = bunch_y #(bunch_y - transverse_axis.min()) / (transverse_axis.max() - transverse_axis.min()) * 2 - 1  # normalize for plotting purposes
        rng = np.random.default_rng()
        idx = rng.choice(bunch_x.shape[0], 2_000, replace=False)
        bunch_x_sampled = bunch_x[idx]
        bunch_y_normalized_sampled = bunch_y_normalized[idx]
        sc = ax.scatter(bunch_x_sampled, bunch_y_normalized_sampled, facecolors='none', edgecolors="cyan", linewidths=0.2, s=0.8)

    #ax.axis('off')
except Exception as error:
    print("Problem in the transverse bunch profile & bunching factor evolution: ", error)


from matplotlib.patches import Rectangle, ConnectionPatch

#ax.set_xlabel(r"z ($\mathrm{\mu m}$)", fontsize=18)
ax.set_ylabel(r"x ($\mathrm{\mu m}$)", fontsize=18, labelpad=10)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_ylim(bottom=transverse_axis_normalized[0], top=transverse_axis_normalized[-1])

#ax.set_xticks([600, 1000, 1400])
ax.set_yticks([120, 140, 160])
#ax.set_xticklabels(["600", "1000", "1400"], fontsize=18)
ax.set_yticklabels(["120", "140", "160"], fontsize=18)
#ax.tick_params(axis="x", which="major", labelsize=18, length=8, width=1.5, direction="out")
ax.tick_params(axis="x", which="both", direction="out", labelsize=18, length=8, width=1.5,  labelbottom=False)
ax.sharex(axs["top2"])

ax.text(0.70, 0.86, r"saturation regime", fontfamily="sans-serif", fontsize=15, color="white", fontweight="bold", transform=ax.transAxes)
ax.text(0.40, 0.86, r"high-gain regime", fontfamily="sans-serif", fontsize=15, color="white", fontweight="bold", transform=ax.transAxes)
ax.text(0.10, 0.86, r"undulator entrance", fontfamily="sans-serif", fontsize=15, color="white", fontweight="bold", transform=ax.transAxes)
ax.text(0.14, 0.13, r"$\mathrm{3^{rd} \,\, slice}$", fontfamily="sans-serif", fontsize=15, color="cyan", fontweight="bold", transform=ax.transAxes)
ax.text(0.38, 0.13, r"$\mathrm{2^{nd} \,\, slice}$", fontfamily="sans-serif", fontsize=15, color="cyan", fontweight="bold", transform=ax.transAxes)
ax.text(0.62, 0.13, r"$\mathrm{1^{st} \,\, slice}$", fontfamily="sans-serif", fontsize=15, color="cyan", fontweight="bold", transform=ax.transAxes)

ax.text(0.01, 0.03, r"$\mathrm{t_{sim} = 1.6\,ps}$", fontfamily="sans-serif", fontsize=14, color="coral", fontweight="bold", transform=ax.transAxes)

ax.text(0.10, 1.02, r"a", fontfamily="sans-serif", fontsize=22, fontweight="bold", transform=ax.transAxes)




####################################   Temporal spike structure for multi spike (different regimes)   #####################################################################################################
ax = axs["top2"]
lws = [2.5, 2.5]
linecolors = ["whitesmoke", "lime"]

timesteps = [97216]
left_x_crop_percentages = [0.20]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
right_x_crop_percentages = [0.85]
up_y_crop_percentages = [0.40]
down_y_crop_percentages = [0.60]
norm_value = 1
line_plots = []
sc = None

simulation_results_path_index = 0
simulation_results_path = simulation_results_paths[0]
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
    print("simulation time at this timestep (in sec.)" , timesteps[0] * dt / wr)
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
    prop_sign = (FieldMatrix * Bz - Ez * By) < 0  # negative propagation which will be filtered later
    labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
    labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
    labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
    labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
    poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0
    poynting_flux_amplitude[prop_sign] = 0


    #print("Field data shape at a specific time --> ", FieldMatrix.shape)
    FieldMatrix = poynting_flux_amplitude   ## assign matrix to the imaging field
    FieldMatrix = np.rot90(FieldMatrix) # proper rotation
    x_axis = TargetFieldDiag.getAxis("x")
    transverse_axis = TargetFieldDiag.getAxis(transverse_axis_name)

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
    transverse_axis = transverse_axis[x_bound_min:x_bound_max]
    x_diff = x_axis[-1] - x_axis[0]
    y_diff = transverse_axis[-1] - transverse_axis[0]
    transverse_axis_normalized = transverse_axis #(transverse_axis - transverse_axis.min()) / (transverse_axis.max() - transverse_axis.min()) * 2 - 1  # normalize for plotting purposes

    im = ax.imshow(
        FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], cmap="inferno", extent=[x_axis[0], x_axis[-1], transverse_axis_normalized[0], transverse_axis_normalized[-1]],
                   aspect="auto", norm=SymLogNorm( np.abs(np.max(FieldMatrix)) / 250)
    )
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    import matplotlib.ticker as mticker
    ticks = [1e21, 1e22, 1e23]
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    cb.set_ticks(ticks)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
    cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
    cb.set_label(r"$\mathrm{W\,/\,m^{2}}$", fontsize=18, labelpad=10)

    diagnostics_axes = ["moving_x", "x", "y", "z", "w"]
    species_list = ["electron", "electron_1", "electron_2"]
    for species in species_list:
        ParticleTrackDiag = S.TrackParticles(species=species, timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        indexes = np.arange(0, len(weights), dtype=int)  # default indexes
        bunch_x = track_data["x"][indexes]
        bunch_y = track_data[transverse_axis_name][indexes]
        bunch_y_normalized = bunch_y #(bunch_y - transverse_axis.min()) / (transverse_axis.max() - transverse_axis.min()) * 2 - 1  # normalize for plotting purposes
        rng = np.random.default_rng()
        idx = rng.choice(bunch_x.shape[0], 2_000, replace=False)
        bunch_x_sampled = bunch_x[idx]
        bunch_y_normalized_sampled = bunch_y_normalized[idx]
        sc = ax.scatter(bunch_x_sampled, bunch_y_normalized_sampled, facecolors='none', edgecolors="cyan", linewidths=0.2, s=0.8)

    #ax.axis('off')
except Exception as error:
    print("Problem in the transverse bunch profile & bunching factor evolution: ", error)


from matplotlib.patches import Rectangle, ConnectionPatch
# --- draw a rectangle in ax1 (axes coords) ---
x0, y0, w, h = 0.40, 0.23, 0.31, 0.53
rect = Rectangle((x0, y0), w, h, transform=ax.transAxes,
                 fill=False, edgecolor='crimson', linewidth=3)
ax.add_patch(rect)
# rectangle lower corners in ax1's axes coords
p_left  = (x0, y0)
p_right = (x0 + w, y0)
# target points: top-left and top-right edge of ax2 (axes coords)
q_left  = (0.0, 1.0)
q_right = (1.0, 1.0)
# --- connect them with lines ---
conn1 = ConnectionPatch(p_left, q_left,
                        coordsA=ax.transAxes, coordsB=axs["top3"].transAxes,
                        axesA=ax, axesB=axs["top3"],
                        color='crimson', linewidth=4, linestyle="--", zorder=10)
conn2 = ConnectionPatch(p_right, q_right,
                        coordsA=ax.transAxes, coordsB=axs["top3"].transAxes,
                        axesA=ax, axesB=axs["top3"],
                        color='crimson', linewidth=4, linestyle="--", zorder=10)

fig.add_artist(conn1)
fig.add_artist(conn2)


ax.set_xlabel(r"z ($\mathrm{\mu m}$)", fontsize=18)
ax.set_ylabel(r"x ($\mathrm{\mu m}$)", fontsize=18, labelpad=10)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_ylim(bottom=transverse_axis_normalized[0], top=transverse_axis_normalized[-1])

ax.set_xticks([500, 1000, 1500])
ax.set_yticks([120, 140, 160])
ax.set_xticklabels(["500", "1000", "1500"], fontsize=18)
ax.set_yticklabels(["120", "140", "160"], fontsize=18)
ax.tick_params(axis="x", which="major", labelsize=18, length=8, width=1.5, direction="out")


ax.text(0.80, 0.86, r"undulator exit", fontfamily="sans-serif", fontsize=15, color="white", fontweight="bold", transform=ax.transAxes)
ax.text(0.47, 0.86, r"post-saturation", fontfamily="sans-serif", fontsize=15, color="white", fontweight="bold", transform=ax.transAxes)
ax.text(0.13, 0.86, r"high-gain regime", fontfamily="sans-serif", fontsize=15, color="white", fontweight="bold", transform=ax.transAxes)
ax.text(0.80, 0.13, r"$\mathrm{1^{st} \,\, slice}$", fontfamily="sans-serif", fontsize=15, color="cyan", fontweight="bold", transform=ax.transAxes)
ax.text(0.40, 0.13, r"$\mathrm{2^{nd} \,\, slice}$", fontfamily="sans-serif", fontsize=15, color="cyan", fontweight="bold", transform=ax.transAxes)
ax.text(0.15, 0.13, r"$\mathrm{3^{rd} \,\, slice}$", fontfamily="sans-serif", fontsize=15, color="cyan", fontweight="bold", transform=ax.transAxes)

ax.text(0.01, 0.03, r"$\mathrm{t_{sim} = 3.1\,ps}$", fontfamily="sans-serif", fontsize=14, color="coral", fontweight="bold", transform=ax.transAxes)


ax.text(0.10, 1.02, r"b", fontfamily="sans-serif", fontsize=22, fontweight="bold", transform=ax.transAxes)


####################################   Temporal spike structure at the exit (single slice)   #####################################################################################################
ax = axs["top3"]
lws = [2.5, 2.5]
linecolors = ["whitesmoke", "lime"]

timesteps = [97216, 97216]
left_x_crop_percentages = [0.40, 0.40]   ## should be close to longitudinal bunch boundaries as mostly the structure within the bunch should be examined
right_x_crop_percentages = [0.70, 0.70]
up_y_crop_percentages = [0.40, 0.40]
down_y_crop_percentages = [0.60, 0.60]
norm_value = 1
line_plots = []
sc = None
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
        print("simulation time at this timestep (in sec.)" , timesteps[0] * dt / wr)
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
        prop_sign = (FieldMatrix * Bz - Ez * By) < 0  # negative propagation which will be filtered later
        labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
        labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
        labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
        labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
        poynting_flux_amplitude = (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0
        poynting_flux_amplitude[prop_sign] = 0


        #print("Field data shape at a specific time --> ", FieldMatrix.shape)
        FieldMatrix = poynting_flux_amplitude   ## assign matrix to the imaging field
        FieldMatrix = np.rot90(FieldMatrix) # proper rotation
        x_axis = TargetFieldDiag.getAxis("x")
        transverse_axis = TargetFieldDiag.getAxis(transverse_axis_name)

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
        transverse_axis = transverse_axis[x_bound_min:x_bound_max]
        x_diff = x_axis[-1] - x_axis[0]
        y_diff = transverse_axis[-1] - transverse_axis[0]
        transverse_axis_normalized = (transverse_axis - transverse_axis.min()) / (transverse_axis.max() - transverse_axis.min()) * 2 - 1  # normalize for plotting purposes
        if simulation_results_path_index == 0:  # ep case
            im = ax.imshow(
                FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], cmap="inferno", extent=[x_axis[0], x_axis[-1], transverse_axis_normalized[0], transverse_axis_normalized[-1]],
                           aspect="auto", norm=SymLogNorm( np.abs(np.max(FieldMatrix)) / 250)
            )
            cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
            import matplotlib.ticker as mticker
            ticks = [1e21, 1e22, 1e23]
            cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
            cb.set_ticks(ticks)
            cb.update_ticks()
            cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
            cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
            cb.set_label(r"$\mathrm{W\,/\,m^{2}}$", fontsize=18, labelpad=10)

        up_y_crop_percentage = 0.45  # for the centralized pulse amplitude, reflections avoided
        down_y_crop_percentage = 0.55
        x_bound_min = int(FieldMatrix.shape[0] * up_y_crop_percentage)
        x_bound_max = int(FieldMatrix.shape[0] * down_y_crop_percentage)
        normalized_y = np.sum(FieldMatrix[x_bound_min:x_bound_max, y_bound_min:y_bound_max], axis=0)
        #kernel_size = 20   ### Convolve the pulse
        #kernel = np.ones(kernel_size) / kernel_size
        #normalized_y = np.convolve(normalized_y, kernel, mode='same')
        from scipy.ndimage import maximum_filter1d
        normalized_y = maximum_filter1d(normalized_y, size=25, mode='nearest')    # for smooth envelope profile visualization purposes
        if simulation_results_path_index == 0:  # ep case
            norm_value = np.max(normalized_y)
            normalized_y /= norm_value
        else:
            normalized_y /= norm_value

        if simulation_results_path_index == 0:  # ep case
            diagnostics_axes = ["moving_x", "x", "y", "z", "w"]
            ParticleTrackDiag = S.TrackParticles(species="electron_1", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
            track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
            weights = track_data["w"]
            indexes = np.arange(0, len(weights), dtype=int)  # default indexes
            bunch_x = track_data["x"][indexes]
            bunch_y = track_data[transverse_axis_name][indexes]
            bunch_y_normalized = (bunch_y - transverse_axis.min()) / (transverse_axis.max() - transverse_axis.min()) * 2 - 1  # normalize for plotting purposes
            rng = np.random.default_rng()
            idx = rng.choice(bunch_x.shape[0], 5_000, replace=False)
            bunch_x_sampled = bunch_x[idx]
            bunch_y_normalized_sampled = bunch_y_normalized[idx]
            sc = ax.scatter(bunch_x_sampled, bunch_y_normalized_sampled, facecolors='none', edgecolors="cyan", linewidths=0.3, s=1.0,
                            label=r"pair beam ($\mathrm{\rho \approx 1.5\!\times\!10^{-3}, \, \sigma_{\delta} = 0.03\%, \, \epsilon_{n} = 0.1\,\mu m\text{-}rad}$)")


        (l1, ) = ax.plot(x_axis , normalized_y,
                        lw=lws[simulation_results_path_index], c=linecolors[simulation_results_path_index], label=labels[simulation_results_path_index])
        line_plots.append(l1)



        #ax.axis('off')
    except Exception as error:
        print("Problem in the transverse bunch profile & bunching factor evolution: ", error)

legend_lines = ax.legend(handles=[line_plots[0], line_plots[1]], fontsize=18, loc=(0.80, 0.65), frameon=False, markerscale=18, labelcolor='linecolor')
ax.add_artist(legend_lines)
legend_scatter = ax.legend(handles=[sc], fontsize=15, loc='lower right', frameon=False, markerscale=9, labelcolor='cyan')
h = legend_scatter.legendHandles[0]                   # PathCollection for the scatter
ec = h.get_edgecolor()
if hasattr(ec, '__len__'): ec = ec[0]      # handle array-like colors
h.set_facecolor(ec)                        # fill with edge color
h.set_alpha(1) 

ax.set_xlabel(r"z ($\mathrm{\mu m}$)", fontsize=18)
ax.set_ylabel("intensity [a.u.]", fontsize=18, labelpad=35)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_ylim(bottom=transverse_axis_normalized[0], top=transverse_axis_normalized[-1])
ax.set_xlim(left=840, right=1225) # limit the region of available data

ax.set_xticks([900, 1050, 1200])
ax.set_yticks([0.25, 0.50, 0.75])
ax.set_xticklabels(["900", "1050", "1200"], fontsize=18)
ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=18)
ax.tick_params(axis="x", which="major", labelsize=18, length=8, width=1.5, direction="out")
#ax.tick_params(axis="y", which="minor", labelsize=18, length=8, width=1.5, direction="out")
ax.tick_params(axis="y", which="major", labelsize=18, length=8, width=1.5, direction="in", colors="white", pad=-50)

FWHM_start_end = [1015, 1110] # in um
lab_frame_pulse_FWHM = (FWHM_start_end[-1] - FWHM_start_end[0]) / (2 * reference_frame_gamma) * 1e-6 / c * 1e18  ## in atto-seconds
print("lab-frame pulse FWHM (as) expected: ", lab_frame_pulse_FWHM)

ax.annotate(
    "", xy=(FWHM_start_end[-1], 0.49), xytext=(FWHM_start_end[0], 0.49),
    arrowprops=dict(arrowstyle="<->", lw=4, mutation_scale=19, color="goldenrod")  # width & head size
)
ax.text(0.495, 0.77, r"$\mathbf{\tau_{lab}\, \approx \, 3.5 \, as}$", fontfamily="sans-serif", fontsize=18, color="goldenrod", fontweight="bold", transform=ax.transAxes)


ax.text(0.01, 0.03, r"$\mathrm{t_{sim} = 3.1\,ps}$", fontfamily="sans-serif", fontsize=14, color="coral", fontweight="bold", transform=ax.transAxes)


ax.text(0.10, 1.02, r"c", fontfamily="sans-serif", fontsize=22, fontweight="bold", transform=ax.transAxes)


######################################       RaDiO      ########################################################################################################################

ax = axs["bottom"][0]

labels = [r"$e^- \, / \,\, e^+$", r"$e^-$"]
colors = ["coral", "royalblue"]

fill_color = [False, True]
linewidths = [3., 3.]

collected_spectra = []
freqs = []
collected_freqs = []

for simulation_results_path_index, simulation_results_path in enumerate(simulation_results_paths):
    RaDiO_fields_dict = {"filenames": [],
                     "deposited_fields": {
                         "ranks": [],
                         "date_times": [],
                         "RaDiO_fields": [],
                     }
                     }

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
                    logging.info("Radiation Detector read --> file: {} , rank: {}, datetime: {}".format(os.path.basename(radiation_file), rank, date_time))
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
        # time evolution of the electric fields
        signal_time = collected_RaDiO_field.time_array[1:] * 1. / wr   # in sec.
        window_start = int(len(signal_time) * 0.10) # 0
        window_end = int(len(signal_time) * 0.99) # len(signal_time)
        T = signal_time[-1] - signal_time[0]  # total signal elapsed time on the detector
        #axs[0].set_yscale("log")
        # Discrete Fourier Transform, power spectral density, Parseval's theorem check
        signal_amplitude = math.sqrt(power_per_solid_angle_correction) * collected_RaDiO_field.E_y[reference_pixel_index][window_start:window_end]   # Electric field magnitude (considering only the dominant component; plane-wave apprx. will bring multiplication factor of 2 for the spectral density)
        T = signal_time[window_end] - signal_time[window_start]
        N = len(signal_amplitude)
        HF_limit = int(N/10)    # for visualization purposes
        fourierTransform = np.fft.fft(signal_amplitude)/N           # Discrete Fourier Transform and its amplitude normalization
        fourierTransform = fourierTransform[range(int(N/2))]        # exclude sampling frequency
        frequencies      = np.arange(int(N/2)) / T
        power_spectral_density = 2 * np.square(np.abs(fourierTransform))   # d2I/dAdw
        # plot power spectrum with the relativistically Doppler corrected frequencies to get lab-frame reference (assume on-axis pixel currently, off-axis will be checked later)
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        power_spectral_density[0:HF_limit] = np.convolve(power_spectral_density[0:HF_limit], kernel, mode='same')
        ax.plot(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0] / 1000,
                 power_spectral_density[0:HF_limit],
                 c=colors[simulation_results_path_index], linewidth=linewidths[simulation_results_path_index],  label=labels[simulation_results_path_index]) # Normalized PSD with HF-truncated
        if fill_color[simulation_results_path_index]:
            ax.fill_between(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0] / 1000,
                             power_spectral_density[0:HF_limit], 0,
                             color=colors[simulation_results_path_index],
                             alpha=.5)
        np.set_printoptions(threshold=np.inf)
        print(simulation_results_path)
        collected_spectra.append(power_spectral_density[0:HF_limit])
        freqs = frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0] / 1000
        collected_freqs.append(freqs)
        #plt.show(block=True)
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
        logging.info("Parseval (DFT) check --> parseval_t: {} , parseval_w: {} , fraction: {}".format(parseval_t, parseval_w, parseval_t/parseval_w))


main_freq = freqs[np.argmax(np.array(collected_spectra[0]))]
print("w_0 (in keV) --> ", main_freq)

ax.legend(fontsize=15, loc=(0.51, 0.63), frameon=False, markerscale=6)
ax.set_ylabel(r"$d\varepsilon / d\omega d\Omega$ [a.u.]", fontsize=18)
ax.set_xlim(0.7 * main_freq, 5.5*main_freq)
ax.set_ylim(5e20, 4e27)
ax.set_xscale("linear")
ax.set_yscale("log")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider  = make_axes_locatable(ax)
ax_ratio = divider.append_axes("bottom", size="30%", pad=0.12, sharex=ax)
plt.setp(ax.get_xticklabels(), visible=False)  # hide top panel x labels
ax_ratio.spines['top'].set_visible(False)
ax_ratio.set_ylabel("ratio", fontsize=18)
ax_ratio.set_xlabel(r"keV", fontsize=18)  # reuse label if you already set i
ax_ratio.axhline(1.0, lw=0.8, alpha=0.6)
ax_ratio.set_xscale(ax.get_xscale())
ax_ratio.set_yscale("log")


# freq: (N,), amp: (N,), targets: (M,)
rms_freq = 0.05
R = np.ones(len(collected_spectra[0]))
targets = np.array([main_freq, 3*main_freq, 5*main_freq])
freqs = np.array(collected_freqs[0])
idx_0 = np.array([
    (i := np.where((freqs >= t * (1 - rms_freq) ) & (freqs <= t * (1 + rms_freq)))[0])[np.argmax(collected_spectra[0][i])] ###   Focus only for odd harmonics, suppress others
    for t in targets
])
val_0 = collected_spectra[0][idx_0]
freqs = np.array(collected_freqs[1])
idx_1 = np.array([
    (i := np.where((freqs >= t * (1 - rms_freq) ) & (freqs <= t * (1 + rms_freq)))[0])[np.argmax(collected_spectra[1][i])] ###   Focus only for odd harmonics, suppress others
    for t in targets
])
val_1 = collected_spectra[1][idx_1]
R[idx_0] = val_0 / val_1
print(val_0 / val_1)
print(idx_0, len(R))
ax_ratio.plot(collected_freqs[0], R, lw=3, color="goldenrod")

ax_ratio.set_xticks([175, 525, 875])
ax_ratio.set_yticks([1, 10, 100, 1000])
#ax.set_yticks([1e22, 1e24, 1e26])
ax_ratio.set_xticklabels(["175", "525", "875"], fontsize=18)
ax_ratio.set_yticklabels([r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"], fontsize=10)
#ax.set_yticklabels([r"$10^{22}$", r"$10^{24}$", r"$10^{26}$"], fontsize=24)
ax.tick_params(axis="y", which="major", direction="out", labelsize=16, length=7, width=1.3)
ax_ratio.tick_params(axis="x", which="major", direction="out", labelsize=18, length=8, width=1.5)
ax_ratio.tick_params(axis="y", which="major", direction="out", labelsize=15, length=8, width=1.5)



ax.text(0.1, 0.82, r"n = 1", fontfamily="sans-serif", fontsize=14, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.51, 0.48, r"n = 3", fontfamily="sans-serif", fontsize=14, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.80, 0.42, r"n = 5", fontfamily="sans-serif", fontsize=14, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
#ax.text(0.91, 0.54, r"n = 7", fontfamily="sans-serif", fontsize="large", fontstyle="italic", fontweight="semibold", transform=ax.transAxes)



ax.text(0.02, 1.04, r"d", fontfamily="sans-serif", fontsize=22, fontweight="bold", transform=ax.transAxes)

##################################          Poynting flux distribution image over plate         ####################################################################################

ax = axs["bottom"][2]

saturation_power_distributions = []
simulation_results_path_index = 1
simulation_results_path = simulation_results_paths[simulation_results_path_index]

saturation_power_distribution = None
sat_power = 0
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

    ## init data retrieval
    timestep = int(92008)   # should be around saturation

    subset = None
    timesteps = S.Field(2, field="Ey", units=unit_dictionary["units"]).getTimesteps().astype(int)
    print("Timesteps for the transverse field screen data --> ", len(timesteps), timesteps)
    total_radiated_power = []
    relative_undulator_pass_distance = []
    initialized_value = 0
    initial_step_passed = False
    transverse_box_shape = [S.namelist.Main.number_of_cells[1]]
    if "3D" in simulation_geometry:
        transverse_box_shape = [S.namelist.Main.number_of_cells[1], S.namelist.Main.number_of_cells[2]]
    poynting_flux_y_limits = [int(0.44 * transverse_box_shape[0]), int(0.62 * transverse_box_shape[0])]
    if "3D" in simulation_geometry:
        poynting_flux_z_limits = [int(0.42 * transverse_box_shape[1]), int(0.60 * transverse_box_shape[1])]

    TargetFieldDiag = S.Field(2, field="Ey", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    TargetDiag_Ez = S.Field(2, field="Ez", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    TargetDiag_By = S.Field(2, field="By_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    TargetDiag_Bz = S.Field(2, field="Bz_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
    Ez = np.array(TargetDiag_Ez.getData())[0]
    By = np.array(TargetDiag_By.getData())[0]
    Bz = np.array(TargetDiag_Bz.getData())[0]
    labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
    labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
    labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
    labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
    poynting_flux_amplitude = np.array( (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0 )
    poynting_flux_amplitude[poynting_flux_amplitude < 0] = 1e-8

    if "3D" in simulation_geometry: # surface integral
        total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr * S.namelist.Main.cell_length[2] * c / wr # in Watt
    else: # line integral
        total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr  # in Watt
    # get the relative distance w.r.t. undulator entrance in lab frame
    bunch_end_boosted_frame = np.max( S.TrackParticles(species="electron", timesteps=timestep, axes=["x"], sort=False, units=unit_dictionary["units"]).getData()[timestep]["x"] )  # in um
    relative_undulator_pass = reference_frame_gamma * (bunch_end_boosted_frame / 1e6 + reference_frame_velocity * c * timestep * dt * wr ** (-1) ) - simulation_parameters["undulator_start_lab_frame"] * c / wr  # in meter
    if relative_undulator_pass > 0:  # for visualization of the radiated power w.r.t. undulator distance in S.I. units (meter currently)
        total_radiated_power.append(np.log10(total_power/2))
        relative_undulator_pass_distance.append(relative_undulator_pass)
    print("Total radiated power --> ", np.log10(total_power/2) )
    # poynting flux
    transverse_axis = TargetFieldDiag.getAxis("y")

    full_size = int(poynting_flux_amplitude.shape[0])
    print(full_size)
    im = ax.imshow(
        poynting_flux_amplitude[poynting_flux_y_limits[0] : poynting_flux_y_limits[1], poynting_flux_z_limits[0] : poynting_flux_z_limits[1]],
        cmap="hot",
        extent=[transverse_axis[poynting_flux_y_limits[0]], transverse_axis[poynting_flux_y_limits[1]], transverse_axis[poynting_flux_z_limits[0]], transverse_axis[poynting_flux_z_limits[1]]],
        aspect="auto",
    )

    
    #cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    #cb.ax.locator_params(nbins=3)
    #cb.update_ticks()
    #cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
    #cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
    #cb.set_label(r"$\mathrm{W\,/\,m^{2}}$", fontsize=18)

    #from matplotlib.ticker import ScalarFormatter
    #fmt = ScalarFormatter(useMathText=True)
    #fmt.set_powerlimits((0, 0))     # always use scientific with exponent
    #cb.formatter = fmt
    #cb.update_ticks()

    #offset = cb.ax.yaxis.get_offset_text()
    #offset.set_fontsize(20)         # or offset.set_size(12)
    #offset.set_fontweight('bold')   # optional
    
    caxR = inset_axes(ax, width="3%", height="99%",
                  loc="upper right", borderpad=0.03)
    caxR.set_in_layout(False)
    cb = fig.colorbar(im, cax=caxR)
    cb.ax.locator_params(nbins=3)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
    cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
    cb.set_label(r"$\mathrm{W\,/\,m^{2}}$", fontsize=18)

    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))     # always use scientific with exponent
    cb.formatter = fmt
    cb.update_ticks()

    offset = cb.ax.yaxis.get_offset_text()
    offset.set_fontsize(20)         # or offset.set_size(12)
    offset.set_fontweight('bold')   # optional
    


except Exception as e:
    print(e)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xlabel(r"x ($\mathrm{\mu m}$)", fontsize=18)
#ax.set_ylabel(r"y ($\mu m$)", fontsize=18, labelpad=1)
ax.set_xticks([140, 160])
ax.set_yticks([140, 160])
#ax.set_xticklabels(["60", "75", "90"], fontsize=18)
#ax.set_yticklabels([], fontsize=18)
ax.tick_params(axis="both", which="major", labelsize=18, length=8, width=1.5, direction="out")
ax.tick_params(axis="y", which="both", labelleft=False)

ax.text(0.20, 0.12, r"($\mathrm{e^{-}-}$only beam)", fontfamily="sans-serif", fontsize=16, color="white", fontweight="semibold", transform=ax.transAxes)


ax.text(0.02, 1.04, r"f", fontfamily="sans-serif", fontsize=22, fontweight="bold", transform=ax.transAxes)

##################################          Poynting flux distribution image over plate         ####################################################################################

ax = axs["bottom"][1]

saturation_power_distributions = []
simulation_results_path_index = 0
simulation_results_path = simulation_results_paths[simulation_results_path_index]

saturation_power_distribution = None
sat_power = 0
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

    ## init data retrieval
    timestep = int(93744)   # should be around saturation

    subset = None
    timesteps = S.Field(2, field="Ey", units=unit_dictionary["units"]).getTimesteps().astype(int)
    print("Timesteps for the transverse field screen data --> ", len(timesteps), timesteps)
    total_radiated_power = []
    relative_undulator_pass_distance = []
    initialized_value = 0
    initial_step_passed = False
    transverse_box_shape = [S.namelist.Main.number_of_cells[1]]
    if "3D" in simulation_geometry:
        transverse_box_shape = [S.namelist.Main.number_of_cells[1], S.namelist.Main.number_of_cells[2]]
    poynting_flux_y_limits = [int(0.44 * transverse_box_shape[0]), int(0.62 * transverse_box_shape[0])]
    if "3D" in simulation_geometry:
        poynting_flux_z_limits = [int(0.42 * transverse_box_shape[1]), int(0.60 * transverse_box_shape[1])]

    TargetFieldDiag = S.Field(2, field="Ey", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    TargetDiag_Ez = S.Field(2, field="Ez", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    TargetDiag_By = S.Field(2, field="By_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    TargetDiag_Bz = S.Field(2, field="Bz_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
    FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
    Ez = np.array(TargetDiag_Ez.getData())[0]
    By = np.array(TargetDiag_By.getData())[0]
    Bz = np.array(TargetDiag_Bz.getData())[0]
    labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
    labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
    labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
    labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
    poynting_flux_amplitude = np.array( (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0 )
    poynting_flux_amplitude[poynting_flux_amplitude < 0] = 1e-8

    if "3D" in simulation_geometry: # surface integral
        total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr * S.namelist.Main.cell_length[2] * c / wr # in Watt
    else: # line integral
        total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr  # in Watt
    # get the relative distance w.r.t. undulator entrance in lab frame
    bunch_end_boosted_frame = np.max( S.TrackParticles(species="electron", timesteps=timestep, axes=["x"], sort=False, units=unit_dictionary["units"]).getData()[timestep]["x"] )  # in um
    relative_undulator_pass = reference_frame_gamma * (bunch_end_boosted_frame / 1e6 + reference_frame_velocity * c * timestep * dt * wr ** (-1) ) - simulation_parameters["undulator_start_lab_frame"] * c / wr  # in meter
    if relative_undulator_pass > 0:  # for visualization of the radiated power w.r.t. undulator distance in S.I. units (meter currently)
        total_radiated_power.append(np.log10(total_power/2))
        relative_undulator_pass_distance.append(relative_undulator_pass)
    print("Total radiated power --> ", np.log10(total_power/2) )
    # poynting flux
    transverse_axis = TargetFieldDiag.getAxis("y")
    full_size = int(poynting_flux_amplitude.shape[0])
    print(full_size)
    im = ax.imshow(
        poynting_flux_amplitude[poynting_flux_y_limits[0] : poynting_flux_y_limits[1], poynting_flux_z_limits[0] : poynting_flux_z_limits[1]],
        cmap="hot",
        extent=[transverse_axis[poynting_flux_y_limits[0]], transverse_axis[poynting_flux_y_limits[1]], transverse_axis[poynting_flux_z_limits[0]], transverse_axis[poynting_flux_z_limits[1]]],
        aspect="auto",
    )
    
    #cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    #cb.ax.locator_params(nbins=3)
    #cb.update_ticks()
    #cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
    #cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
    ##cb.set_label(r"$W\,/\,m^{2}$", fontsize=18)

    #from matplotlib.ticker import ScalarFormatter
    #fmt = ScalarFormatter(useMathText=True)
    #fmt.set_powerlimits((0, 0))     # always use scientific with exponent
    #cb.formatter = fmt
    #cb.update_ticks()

    #offset = cb.ax.yaxis.get_offset_text()
    #offset.set_fontsize(20)         # or offset.set_size(12)
    #offset.set_fontweight('bold')   # optional
    

    caxM = inset_axes(ax, width="3%", height="99%",
                  loc="upper right", borderpad=0.05)
    caxM.set_in_layout(False)
    cb = fig.colorbar(im, cax=caxM, pad = 0.03)
    cb.ax.locator_params(nbins=3)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=18)   # set tick LABEL font size
    cb.ax.tick_params(length=7, width=1.2)  # (optional) tick mark size/width
    #cb.set_label(r"$W\,/\,m^{2}$", fontsize=18)
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))     # always use scientific with exponent
    cb.formatter = fmt
    cb.update_ticks()

    offset = cb.ax.yaxis.get_offset_text()
    offset.set_fontsize(20)         # or offset.set_size(12)
    offset.set_fontweight('bold')   # optional
    


except Exception as e:
    print(e)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xlabel(r"x ($\mathrm{\mu m}$)", fontsize=18)
ax.set_ylabel(r"y ($\mathrm{\mu m}$)", fontsize=18, labelpad=4)
ax.set_xticks([140, 160])
ax.set_yticks([140, 160])
#ax.set_xticklabels(["60", "75", "90"], fontsize=18)
#ax.set_yticklabels(["60", "75", "90"], fontsize=18)
ax.tick_params(axis="both", which="major", labelsize=18, length=8, width=1.5, direction="out")

ax.text(0.25, 0.82, r"$\mathbf{P_{sat} \, \approx\, 10 \, TW}$", fontfamily="sans-serif", fontsize=18, color="white", fontstyle="italic", fontweight="bold", transform=ax.transAxes)

ax.text(0.24, 0.12, r"(pair beam)", fontfamily="sans-serif", fontsize=16, color="white", fontweight="semibold", transform=ax.transAxes)


ax.text(0.02, 1.04, r"e", fontfamily="sans-serif", fontsize=22, fontweight="bold", transform=ax.transAxes)

###############################################################################################################################################################################
#plt.tight_layout()
pathlib.Path(figname + ".png").unlink(missing_ok=True)
plt.savefig(figname + ".png", format="png", bbox_inches="tight", dpi=200)
plt.close(fig)


