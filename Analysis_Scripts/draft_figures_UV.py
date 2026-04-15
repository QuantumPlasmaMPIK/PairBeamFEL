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
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

########################################################################################################################

labels = [r"$e^- \, / \,\, e^+$", r"$e^-$"]
colors = ["coral", "royalblue"]


reference_frame_gamma = 1
reference_frame_velocity = 0


fig_w = 25
fig_h = fig_w * (2 / 4.60) # width units: 3 (grid cols) + 0.20 (spacer) + 1.1 (right panel) =
fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
fig.set_constrained_layout_pads(h_pad=0.10, w_pad=0.20, wspace=0.06, hspace=0.07)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.10, 1.50])  # col 3 spacer, col 4 right panel

figname = "UV"

####################################    1d distribution of macro-particles    #####################################################################################################

simulation_results_paths = ['/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_UV_ep_pancake_3/simulation_results/',
                            '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_UV_e_pancake_2/simulation_results/',
                            ]


ax = fig.add_subplot(gs[1, 0])
ax.set_box_aspect(1)
#ax.sharex(axs[1][0])
snapshot_timesteps = [21414, 20196]
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

        ## init data retrieval
        diagnostics_axes = ["moving_x", "x", "y", "z", "w"]
        transverse_axis_name = "z"
        subset = None
        tracked_species = S.getTrackSpecies()
        species_labels = [r"$e^+$", r"$e^-$"]
        bunch_timestep = snapshot_timesteps[simulation_results_path_index]   #S.TrackParticles(species=tracked_species[0], axes=["x"], sort=False, units=unit_dictionary["units"]).getTimesteps().astype(int)
        VISUALIZATION_NUMBER_LIMIT = int(1e6)
        species = tracked_species[0] # electron distribution is considered as a default
        # Spatial distribution
        ParticleTrackDiag = S.TrackParticles(species=species, timesteps=bunch_timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[bunch_timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        print("Number of loaded macro-particles for " + species + ": ", weights.shape[0])
        #if weights.shape[0] > VISUALIZATION_NUMBER_LIMIT:
        #    indexes = np.random.choice(weights.shape[0], int(VISUALIZATION_NUMBER_LIMIT), False)
        #else:
        #    indexes = np.arange(0, len(weights), dtype=int)  # default indexes
        #print("Number of visualized macro-particles for " + species + ": ", indexes.shape[0])

        #hist_data = track_data["x"][indexes] - np.min(track_data["x"][indexes])
        hist_data = track_data["x"][:] - np.min(track_data["x"][:])
        bins = np.linspace(np.min(hist_data), np.max(hist_data), 350) # create bin to set the interval
        graph, edges = np.histogram(hist_data, bins, density=True) # create histogram
        bias_value = 0.7e-4   ### for a nice numerical range for plotting purposes
        norm_value = 2.7e-4
        normalized_graph = graph #graph / np.max(graph)
        ax.plot(edges[:-1] / np.max(edges[:-1]), (normalized_graph - bias_value) / norm_value, c=colors[simulation_results_path_index], label=labels[simulation_results_path_index], lw=0.5)
        #ax.axis('off')
    except Exception as error:
        print("Problem in the transverse bunch profile & bunching factor evolution: ", error)


ax.set_ylabel(r"density [a.u.]", fontsize=28)
ax.set_xlabel(r"bunch coord. [a.u.]", fontsize=28)
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))
#ax.yaxis.set_major_locator(plt.MaxNLocator(3))
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_ylim(bottom=0.125, top=1.03)
ax.set_xlim(left=-0.01, right=1.01)
ax.set_xticks([0, 0.5, 1.0])
ax.set_yticks([0.5, 1])
ax.set_xticklabels(["0", "0.5", "1"], fontsize=25)
ax.set_yticklabels([ r"$0.5$", r"$1$"], fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
#ax.minorticks_on()
#ax.tick_params(axis="both", which="minor", labelsize=14, length=3, width=0.9, direction="out")

ax.text(0.05, 0.90, r"d", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


#########################################  Gain Curve - flux distr.    ######### = ###################################################################################################

ax = fig.add_subplot(gs[1, 1])
ax.set_box_aspect(1)
#ax.sharex(axs[1][1])

saturation_power_distributions = []
KE_dists = []
bottom_power_limit = 6.  ## in log() GW
plate_distance = 2.5 ## in m.
time_threshold_indexes = [1, 1] # to get signal until around saturation point
for simulation_results_path_index, simulation_results_path in enumerate(simulation_results_paths):
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
        subset = None
        timesteps = S.Field(1, field="Ey", units=unit_dictionary["units"]).getTimesteps().astype(int)
        print("Timesteps for the transverse field screen data --> ", len(timesteps), timesteps)
        total_radiated_power = []
        relative_undulator_pass_distance = []
        initialized_value = 0
        initial_step_passed = False
        transverse_box_shape = [S.namelist.Main.number_of_cells[1]]
        if "3D" in simulation_geometry:
            transverse_box_shape = [S.namelist.Main.number_of_cells[1], S.namelist.Main.number_of_cells[2]]
        poynting_flux_y_limits = [int(0.40 * transverse_box_shape[0]), int(0.60 * transverse_box_shape[0])]
        if "3D" in simulation_geometry:
            poynting_flux_z_limits = [int(0.40 * transverse_box_shape[1]), int(0.60 * transverse_box_shape[1])]
        time_threshold_index = int(len(timesteps) * time_threshold_indexes[simulation_results_path_index])
        for timestep in timesteps[0:time_threshold_index]:
            TargetFieldDiag = S.Field(1, field="Ey", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Ez = S.Field(1, field="Ez", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_By = S.Field(1, field="By_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Bz = S.Field(1, field="Bz_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
            Ez = np.array(TargetDiag_Ez.getData())[0]
            By = np.array(TargetDiag_By.getData())[0]
            Bz = np.array(TargetDiag_Bz.getData())[0]
            labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
            labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
            labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
            labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
            poynting_flux_amplitude = np.array( (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0 )
            if not initial_step_passed: # initial field value will be used as an offset
                initial_step_passed = True
                initialized_value = np.copy(poynting_flux_amplitude)
                poynting_flux_amplitude[:] = 1 / poynting_flux_amplitude.size
            else:   # subtract the offset value
                poynting_flux_amplitude -= initialized_value
                poynting_flux_amplitude[poynting_flux_amplitude < 0] = 1e-8
            if "3D" in simulation_geometry: # surface integral
                total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr * S.namelist.Main.cell_length[2] * c / wr # in Watt
            else: # line integral
                total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr # in Watt
            if total_power > sat_power:
                sat_power = total_power
                saturation_power_distribution = (poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]] ).flatten()
            # get the relative distance w.r.t. undulator entrance in lab frame
            bunch_end_boosted_frame = np.max( S.TrackParticles(species="electron", timesteps=timestep, axes=["x"], sort=False, units=unit_dictionary["units"]).getData()[timestep]["x"] )  # in um
            relative_undulator_pass = reference_frame_gamma * (bunch_end_boosted_frame / 1e6 + reference_frame_velocity * c * timestep * dt * wr ** (-1) ) - simulation_parameters["undulator_start_lab_frame"] * c / wr  # in meter
            if relative_undulator_pass > 0:  # for visualization of the radiated power w.r.t. undulator distance in S.I. units (meter currently)
                total_radiated_power.append(np.log10(total_power/2))
                relative_undulator_pass_distance.append(relative_undulator_pass - plate_distance)
            print("Total radiated power --> ", np.log10(total_power/2) )
            bunch_diagnostics_axes = [ "x", "y", "z", "px", "py", "pz", "w"]
            ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=bunch_diagnostics_axes, sort=False) # for all enabled species within this tracking diagnostics
            track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
            weights = track_data["w"]
            longitudinal_positions = track_data["x"]  # longitudinal coordinates (boosted frame --> in Smilei units )
            px = track_data["px"]  # momentum (boosted frame)
            py = track_data["py"]
            pz = track_data["pz"]
            bunch_gamma = np.sqrt(1 + px**2 + py**2 + pz**2)  # gamma factors
            vx = px / bunch_gamma  # longitudinal velocity (boosted frame)
            transformed_px = reference_frame_gamma * (px - bunch_gamma * -1 * reference_frame_velocity) # inverse lorentz transformation (from boosted to lab frame)
            transformed_longitudinal_velocities = (vx - -1 * reference_frame_velocity) / (1 - vx * -1 * reference_frame_velocity)
            transformed_times = reference_frame_gamma * (dt * timestep - -1 * reference_frame_velocity * longitudinal_positions)
            KE = np.sqrt(1 + transformed_px**2 + py**2 + pz**2) - 1  # in mc^2
            KE -= simulation_parameters["bunchgamma"] - 1 #np.mean(KE) # relative shift
            print("KE dist. statistics --> ", np.mean(KE), np.std(KE), simulation_results_path_index)
        saturation_power_distributions.append(saturation_power_distribution)
        # smooth the profile to eliminate high-freq. noise elements
        kernel_k = 3
        total_radiated_power = np.convolve(np.pad(total_radiated_power, (kernel_k//2, kernel_k-1-kernel_k//2), mode="reflect"), np.ones(kernel_k)/kernel_k, mode="valid")
        ax.plot(relative_undulator_pass_distance[0:], total_radiated_power[0:], c=colors[simulation_results_path_index], label=labels[simulation_results_path_index], lw=4)

        timestep = timesteps[time_threshold_index-1]
        bunch_diagnostics_axes = [ "x", "y", "z", "px", "py", "pz", "w"]
        ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=bunch_diagnostics_axes, sort=False) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        longitudinal_positions = track_data["x"]  # longitudinal coordinates (boosted frame --> in Smilei units )
        px = track_data["px"]  # momentum (boosted frame)
        py = track_data["py"]
        pz = track_data["pz"]
        bunch_gamma = np.sqrt(1 + px**2 + py**2 + pz**2)  # gamma factors
        vx = px / bunch_gamma  # longitudinal velocity (boosted frame)
        transformed_px = reference_frame_gamma * (px - bunch_gamma * -1 * reference_frame_velocity) # inverse lorentz transformation (from boosted to lab frame)
        transformed_longitudinal_velocities = (vx - -1 * reference_frame_velocity) / (1 - vx * -1 * reference_frame_velocity)
        transformed_times = reference_frame_gamma * (dt * timestep - -1 * reference_frame_velocity * longitudinal_positions)
        KE = np.sqrt(1 + transformed_px**2 + py**2 + pz**2) - 1  # in mc^2
        KE -= simulation_parameters["bunchgamma"] - 1 #np.mean(KE) # relative shift
        KE_dists.append(KE)
        print("KE dist. statistics --> ", np.mean(KE), np.std(KE), simulation_results_path_index)
    except Exception as e:
        print(e)

ax_hist = ax.inset_axes([0.43, 0.48, 0.50, 0.27])
KE = KE_dists[1]  # KE distribution for the e-only scenario
bins = np.linspace(np.mean(KE) - 1.5*np.std(KE), np.mean(KE) + 1.5*np.std(KE), 100) # create bin to set the interval
graph, edges = np.histogram(KE, bins, density=True) # create histogram
normalized_graph = graph / np.max(graph)
mean_KE = np.mean(KE_dists[0]) # Mean KE value for the e/p scenario
mean_KE_index = (np.abs(edges - mean_KE)).argmin()
mean_KE_edge_value = (edges[mean_KE_index] + edges[mean_KE_index+1]) / 2
mean_KE_ref_index = (np.abs(edges - np.mean(KE_dists[1]))).argmin()
mean_KE_edge_ref_value = (edges[mean_KE_ref_index] + edges[mean_KE_ref_index+1]) / 2
zero_KE_index = (np.abs(edges - 0)).argmin()
bar_colors = np.full(len(normalized_graph), "cornflowerblue")
#if mean_KE_index < mean_KE_ref_index:
#    bar_colors[mean_KE_index:mean_KE_ref_index] = colors[0]
#else:
#    bar_colors[zero_KE_index:zero_KE_index] = colors[0]
ax_hist.bar(edges[:-1], normalized_graph, width=np.diff(edges), edgecolor="black", lw=0.2, align="edge", color=bar_colors)

ax_hist.plot([mean_KE_edge_ref_value, mean_KE_edge_ref_value], [0, normalized_graph[mean_KE_ref_index]], color="black", linestyle="--", linewidth="5")
ax_hist.plot([mean_KE_edge_value, mean_KE_edge_value], [0, normalized_graph[mean_KE_index]], color="black", linestyle="--", linewidth="5")
ax_hist.set_xlabel(r"$\mathrm{E_k\, (mc^2)}$",  fontsize=18)
#ax_hist.set_ylabel("normalized amplitude",  fontsize=10)
ax_hist.set_ylim(bottom=0.45)
ax_hist.set_xlim(left=-12, right=12)
ax_hist.set_xticks([-6, 0, 6])
ax_hist.set_yticks([])
ax_hist.set_xticklabels(["-6", "0", "6"], fontsize=18)
#ax_hist.set_yticklabels(["0.5", "1"], fontsize=16)
ax_hist.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, pad=1, labelsize=18)
ax_hist.xaxis.labelpad = -1
#ax_hist.minorticks_on()
#ax_hist.grid(True, which='both', axis='y', ls='--', lw=0.2)
#for spine in ax_hist.spines.values():
#    spine.set_linewidth(1.1)
[ax_hist.spines[s].set_visible(False) for s in ax_hist.spines]
#ax_hist.set_xscale("log")
#ax_hist.set_yticks([0,1])
ax_hist.set_facecolor("0.9"); ax_hist.patch.set_alpha(1); ax_hist.set_zorder(0)


ax.set_xlim(left=1.5, right=8)
ax.set_ylim(bottom=bottom_power_limit, top=10.8)
ax.set_xlabel(r'z (m)', fontsize=28)
ax.set_ylabel(r'$\mathrm{log_{10}P}$ (W)', fontsize=28)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xticks([2.5, 5, 7.5])
ax.set_yticks([6, 8, 10])
ax.set_xticklabels(["2.5", "5", "7.5"], fontsize=25)
ax.set_yticklabels(["6", "8", "10"], fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
#ax.minorticks_on()
#ax.tick_params(axis="both", which="minor", labelsize=14, length=3, width=0.9, direction="out")

ax.text(0.46, 0.66, r"$\mathbf{e/e^{\!+}}$", fontfamily="sans-serif", fontsize=20, color="crimson", transform=ax.transAxes)
ax.text(0.66, 0.74, r"e", fontfamily="sans-serif", fontsize=20, fontweight="bold", color="crimson", transform=ax.transAxes)

ax.text(0.05, 0.90, r"e", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)



######################################       RaDiO      ########################################################################################################################

labels = [r"$e^- \, / \,\, e^+$ (pancake)", r"$e^-$ (pancake)"]

ax = fig.add_subplot(gs[1, 2])
#ax.set_box_aspect(1)
fill_color = [True, True]
linewidths = [1.0, 0.75]

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
        window_start = int(len(signal_time) * 0.015) # 0
        window_end = int(len(signal_time) * 0.80) # len(signal_time)
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
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        power_spectral_density[0:HF_limit] = np.convolve(power_spectral_density[0:HF_limit], kernel, mode='same')
        ax.plot(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0],
                 power_spectral_density[0:HF_limit],
                 c=colors[simulation_results_path_index], linewidth=linewidths[simulation_results_path_index],  label=labels[simulation_results_path_index]) # Normalized PSD with HF-truncated
        if fill_color[simulation_results_path_index]:
            ax.fill_between(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0],
                             power_spectral_density[0:HF_limit], 0,
                             color=colors[simulation_results_path_index],
                             alpha=.5)
        np.set_printoptions(threshold=np.inf)
        print(simulation_results_path)
        collected_spectra.append(power_spectral_density[0:HF_limit])
        freqs = frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0]
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


main_freq = collected_freqs[0][np.argmax(np.array(collected_spectra[0]))]
print("w_0 (in eV) --> ", main_freq)
ax.set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=28)
ax.set_xlim(0.7 * main_freq, 7.5*main_freq)
ax.set_ylim(1e13, 5e20)
ax.set_xscale("linear")
ax.set_yscale("log")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.legend(fontsize="xx-large", loc='upper right', frameon=True, markerscale=120)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider  = make_axes_locatable(ax)
ax_ratio = divider.append_axes("bottom", size="30%", pad=0.12, sharex=ax)
plt.setp(ax.get_xticklabels(), visible=False)  # hide top panel x labels
ax_ratio.spines['top'].set_visible(False)
ax_ratio.set_ylabel("ratio", fontsize=28, labelpad=18)
ax_ratio.axhline(1.0, lw=0.8, alpha=0.6)
ax_ratio.set_xscale(ax.get_xscale())

#R = np.array(collected_spectra[0]) / np.array(collected_spectra[1])
#R[np.round(freqs / main_freq).astype(int) % 2 == 0] = 1  ###   Focus only for odd harmonics, suppress others
#rms_freq = 0.01
#R[np.logical_and(freqs % main_freq > main_freq * rms_freq*0.1, freqs % main_freq < main_freq * (1 - rms_freq)) ] = 1  ###   Focus only for odd harmonics, suppress others

# freq: (N,), amp: (N,), targets: (M,)
rms_freq = 0.05
R = np.ones(len(collected_spectra[0]))
targets = np.array([main_freq, 3*main_freq, 5*main_freq, 7*main_freq])
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
print(val_0, val_1, val_0 / val_1)



ax_ratio.plot(collected_freqs[0], R, lw=3, color="goldenrod")
ax_ratio.set_ylim(bottom=1, top=1e4)
ax_ratio.set_yscale("log")
ax_ratio.set_xlabel(r"eV", fontsize=28)  # reuse label if you already set i


ax_ratio.set_xticks([30, 60, 90, 120])
ax_ratio.set_yticks([1, 1e2, 1e4])
ax_ratio.set_xticklabels(["30", "60", "90", "120"], fontsize=25)
ax_ratio.set_yticklabels([r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"], fontsize=25)
ax.set_yticks([1e14, 1e16, 1e18, 1e20])
ax.set_yticklabels([r"$10^{14}$", r"$10^{16}$", r"$10^{18}$", r"$10^{20}$"], fontsize=25)
ax_ratio.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
ax.tick_params(axis="y", which="major", labelsize=25, length=8, width=1.5, direction="out")



ax_RaDiO_1 = ax_ratio


ax.text(0.07, 0.91, r"n = 1", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.36, 0.60, r"n = 3", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.65, 0.55, r"n = 5", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.83, 0.45, r"n = 7", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)


ax.text(0.15, 0.65, r"f", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


###############################################################################################################################################################################

####################################    1d distribution of macro-particles    #####################################################################################################

simulation_results_paths = ['/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_UV_ep_ellipsoidal/simulation_results/',
                            '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_UV_e_ellipsoidal/simulation_results/',
                            ]


ax = fig.add_subplot(gs[0, 0])
ax.set_box_aspect(1)
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

        ## init data retrieval
        diagnostics_axes = ["moving_x", "x", "y", "z", "w"]
        transverse_axis_name = "z"
        subset = None
        tracked_species = S.getTrackSpecies()
        species_labels = [r"$e^+$", r"$e^-$"]
        bunch_timestep = [22227, 23900]   #near-saturation points
        VISUALIZATION_NUMBER_LIMIT = int(1e6)
        species = tracked_species[0] # electron distribution is considered as a default
        # Spatial distribution
        ParticleTrackDiag = S.TrackParticles(species=species, timesteps=bunch_timestep[simulation_results_path_index], axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[bunch_timestep[simulation_results_path_index]] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        print("Number of loaded macro-particles for " + species + ": ", weights.shape[0])
        bunch_crop_left = int(weights.shape[0] * 0.60)     ### half=-part of the bunch goes to saturation well, hence only that side is considered for correct microbunch efficiency comparison
        bunch_crop_right = int(weights.shape[0] * 0.90)
        hist_data = track_data["x"][bunch_crop_left:bunch_crop_right] - np.min(track_data["x"][bunch_crop_left:bunch_crop_right])
        #indexes = np.random.choice(weights.shape[0], int(VISUALIZATION_NUMBER_LIMIT), False)
        #hist_data = track_data["x"][indexes] - np.min(track_data["x"][indexes])
        bins = np.linspace(np.min(hist_data), np.max(hist_data), 500) # create bin to set the interval
        graph, edges = np.histogram(hist_data, bins, density=True) # create histogram
        bias_value = 1e-5   ### for a nice numerical range for plotting purposes
        norm_value = 2.5e-4
        normalized_graph = graph #graph / np.max(graph)
        ax.plot(edges[:-1] / np.max(edges[:-1]), (normalized_graph - bias_value) / norm_value, c=colors[simulation_results_path_index], label=labels[simulation_results_path_index], lw=0.5)
        #ax.axis('off')
    except Exception as error:
        print("Problem in the transverse bunch profile & bunching factor evolution: ", error)


ax.set_ylabel(r"density [a.u.]", fontsize=28)
ax.set_xlabel(r"bunch coord. [a.u.]", fontsize=28)
#ax.xaxis.set_major_locator(plt.MaxNLocator(3))
#ax.yaxis.set_major_locator(plt.MaxNLocator(3))
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

#ax.set_ylim(bottom=0, top=1.05)
ax.set_xlim(left=-0.01, right=1.01)
ax.set_xticks([0, 0.5, 1.0])
ax.set_yticks([0.5, 1])
ax.set_xticklabels(["0", "0.5", "1"], fontsize=25)
ax.set_yticklabels([r"$0.5$", r"$1$"], fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
#ax.minorticks_on()
#ax.tick_params(axis="both", which="minor", labelsize=14, length=3, width=0.9, direction="out")

ax.text(0.05, 0.90, r"a", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


#########################################  Gain Curve - flux distr.    ######### = ###################################################################################################

ax = fig.add_subplot(gs[0, 1])
ax.set_box_aspect(1)
saturation_power_distributions = []
KE_dists = []
bottom_power_limit = 2.5  ## in log() GW
plate_distance = 3.2 ## in m.
time_threshold_indexes = [0.80, 0.98] # to get signal until around saturation point
for simulation_results_path_index, simulation_results_path in enumerate(simulation_results_paths):
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
        subset = None
        timesteps = S.Field(1, field="Ey", units=unit_dictionary["units"]).getTimesteps().astype(int)
        print("Timesteps for the transverse field screen data --> ", len(timesteps), timesteps)
        total_radiated_power = []
        relative_undulator_pass_distance = []
        initialized_value = 0
        initial_step_passed = False
        transverse_box_shape = [S.namelist.Main.number_of_cells[1]]
        if "3D" in simulation_geometry:
            transverse_box_shape = [S.namelist.Main.number_of_cells[1], S.namelist.Main.number_of_cells[2]]
        poynting_flux_y_limits = [int(0.35 * transverse_box_shape[0]), int(0.65 * transverse_box_shape[0])]
        if "3D" in simulation_geometry:
            poynting_flux_z_limits = [int(0.35 * transverse_box_shape[1]), int(0.65 * transverse_box_shape[1])]
        time_threshold_index = int(len(timesteps) * time_threshold_indexes[simulation_results_path_index])
        for timestep in timesteps:
            TargetFieldDiag = S.Field(1, field="Ey", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Ez = S.Field(1, field="Ez", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_By = S.Field(1, field="By_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            TargetDiag_Bz = S.Field(1, field="Bz_m", timesteps=timestep, subset = subset, units=unit_dictionary["units"])
            FieldMatrix = np.array(TargetFieldDiag.getData())[0]  # list of arrays, so single timestep correspond to 0th index
            Ez = np.array(TargetDiag_Ez.getData())[0]
            By = np.array(TargetDiag_By.getData())[0]
            Bz = np.array(TargetDiag_Bz.getData())[0]
            labframe_Ey = reference_frame_gamma * (FieldMatrix + reference_frame_velocity * c * Bz)
            labframe_Ez = reference_frame_gamma * (Ez - reference_frame_velocity * c * By)
            labframe_By = reference_frame_gamma * (By - reference_frame_velocity * c / c**2 * Ez)
            labframe_Bz = reference_frame_gamma * (Bz + reference_frame_velocity * c / c ** 2 * FieldMatrix)
            poynting_flux_amplitude = np.array( (labframe_Ey * labframe_Bz - labframe_Ez * labframe_By) / mu0 )
            if not initial_step_passed: # initial field value will be used as an offset
                initial_step_passed = True
                initialized_value = np.copy(poynting_flux_amplitude)
                poynting_flux_amplitude[:] = 1 / poynting_flux_amplitude.size
            else:   # subtract the offset value
                poynting_flux_amplitude -= initialized_value
                poynting_flux_amplitude[poynting_flux_amplitude < 0] = 1e-8
            if "3D" in simulation_geometry: # surface integral
                total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr * S.namelist.Main.cell_length[2] * c / wr # in Watt
            else: # line integral
                total_power = np.sum(poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1]]) * S.namelist.Main.cell_length[1] * c / wr # in Watt
            if total_power > sat_power:
                sat_power = total_power
                saturation_power_distribution = (poynting_flux_amplitude[poynting_flux_y_limits[0]:poynting_flux_y_limits[1], poynting_flux_z_limits[0]:poynting_flux_z_limits[1]] ).flatten()
            # get the relative distance w.r.t. undulator entrance in lab frame
            bunch_end_boosted_frame = np.max( S.TrackParticles(species="electron", timesteps=timestep, axes=["x"], sort=False, units=unit_dictionary["units"]).getData()[timestep]["x"] )  # in um
            relative_undulator_pass = reference_frame_gamma * (bunch_end_boosted_frame / 1e6 + reference_frame_velocity * c * timestep * dt * wr ** (-1) ) - simulation_parameters["undulator_start_lab_frame"] * c / wr  # in meter
            if relative_undulator_pass > 0:  # for visualization of the radiated power w.r.t. undulator distance in S.I. units (meter currently)
                total_radiated_power.append(np.log10(total_power/2))
                relative_undulator_pass_distance.append(relative_undulator_pass - plate_distance)
            print("Total radiated power --> ", np.log10(total_power/2) )
            bunch_diagnostics_axes = [ "x", "y", "z", "px", "py", "pz", "w"]
            ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=bunch_diagnostics_axes, sort=False) # for all enabled species within this tracking diagnostics
            track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
            weights = track_data["w"]
            longitudinal_positions = track_data["x"]  # longitudinal coordinates (boosted frame --> in Smilei units )
            px = track_data["px"]  # momentum (boosted frame)
            py = track_data["py"]
            pz = track_data["pz"]
            bunch_gamma = np.sqrt(1 + px**2 + py**2 + pz**2)  # gamma factors
            vx = px / bunch_gamma  # longitudinal velocity (boosted frame)
            transformed_px = reference_frame_gamma * (px - bunch_gamma * -1 * reference_frame_velocity) # inverse lorentz transformation (from boosted to lab frame)
            transformed_longitudinal_velocities = (vx - -1 * reference_frame_velocity) / (1 - vx * -1 * reference_frame_velocity)
            transformed_times = reference_frame_gamma * (dt * timestep - -1 * reference_frame_velocity * longitudinal_positions)
            KE = np.sqrt(1 + transformed_px**2 + py**2 + pz**2) - 1  # in mc^2
            KE -= simulation_parameters["bunchgamma"] - 1 #np.mean(KE) # relative shift
            print("KE dist. statistics --> ", np.mean(KE), np.std(KE), simulation_results_path_index)
            print("momentums --> ", np.mean(np.abs(px)), np.mean(np.abs(py)), np.mean(np.abs(pz)) )
            print("ref. frame vel., bunch gamma, ref. frame gamma, trans. px. -->", reference_frame_velocity, np.mean(np.abs(bunch_gamma)), reference_frame_gamma, np.mean(np.abs(transformed_px)) )
        saturation_power_distributions.append(saturation_power_distribution)
        # smooth the profile to eliminate high-freq. noise elements
        kernel_k = 5
        total_radiated_power = np.convolve(np.pad(total_radiated_power, (kernel_k//2, kernel_k-1-kernel_k//2), mode="reflect"), np.ones(kernel_k)/kernel_k, mode="valid")
        ax.plot(relative_undulator_pass_distance[0:], total_radiated_power[0:], c=colors[simulation_results_path_index], label=labels[simulation_results_path_index], lw=4)

        timestep = timesteps[time_threshold_index-1]
        bunch_diagnostics_axes = [ "x", "y", "z", "px", "py", "pz", "w"]
        ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=bunch_diagnostics_axes, sort=False) # for all enabled species within this tracking diagnostics
        track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
        weights = track_data["w"]
        longitudinal_positions = track_data["x"]  # longitudinal coordinates (boosted frame --> in Smilei units )
        px = track_data["px"]  # momentum (boosted frame)
        py = track_data["py"]
        pz = track_data["pz"]
        bunch_gamma = np.sqrt(1 + px**2 + py**2 + pz**2)  # gamma factors
        vx = px / bunch_gamma  # longitudinal velocity (boosted frame)
        transformed_px = reference_frame_gamma * (px - bunch_gamma * -1 * reference_frame_velocity) # inverse lorentz transformation (from boosted to lab frame)
        transformed_longitudinal_velocities = (vx - -1 * reference_frame_velocity) / (1 - vx * -1 * reference_frame_velocity)
        transformed_times = reference_frame_gamma * (dt * timestep - -1 * reference_frame_velocity * longitudinal_positions)
        KE = np.sqrt(1 + transformed_px**2 + py**2 + pz**2) - 1  # in mc^2
        KE -= simulation_parameters["bunchgamma"] - 1 #np.mean(KE) # relative shift
        KE_dists.append(KE)
        print("KE dist. statistics --> ", np.mean(KE), np.std(KE), simulation_results_path_index)
    except Exception as e:
        print(e)


ax_hist = ax.inset_axes([0.43, 0.15, 0.50, 0.28])
KE = KE_dists[1]  # KE distribution for the e-only scenario
bins = np.linspace(np.mean(KE) - 1.5*np.std(KE), np.mean(KE) + 1.5*np.std(KE), 100) # create bin to set the interval
graph, edges = np.histogram(KE, bins, density=True) # create histogram
normalized_graph = graph / np.max(graph)
mean_KE = np.mean(KE_dists[0]) # Mean KE value for the e/p scenario
mean_KE_index = (np.abs(edges - mean_KE)).argmin()
mean_KE_edge_value = (edges[mean_KE_index] + edges[mean_KE_index+1]) / 2
zero_KE_index = (np.abs(edges - 0)).argmin()
mean_KE_ref_index = (np.abs(edges - np.mean(KE_dists[1]))).argmin()
mean_KE_edge_ref_value = (edges[mean_KE_ref_index] + edges[mean_KE_ref_index+1]) / 2
bar_colors = np.full(len(normalized_graph), "cornflowerblue")
#if mean_KE_index < mean_KE_ref_index:
#    bar_colors[mean_KE_index:mean_KE_ref_index] = colors[0]
#else:
#    bar_colors[mean_KE_ref_index:mean_KE_index] = colors[1]
ax_hist.bar(edges[:-1], normalized_graph, width=np.diff(edges), edgecolor="black", lw=0.2, align="edge", color=bar_colors)
ax_hist.plot([mean_KE_edge_ref_value, mean_KE_edge_ref_value], [0, normalized_graph[mean_KE_ref_index]], color="black", linestyle="--", linewidth="5")
ax_hist.plot([mean_KE_edge_value, mean_KE_edge_value], [0, normalized_graph[mean_KE_index]], color="black", linestyle="--", linewidth="5")
ax_hist.set_xlabel(r"$\mathrm{E_k\, (m\,c^2)}$",  fontsize=18)
#ax_hist.set_ylabel("normalized amplitude",  fontsize=10)
ax_hist.set_ylim(bottom=0.4)
ax_hist.set_xlim(left=-1.5, right=0.80)
ax_hist.set_xticks([ -1, 0])
ax_hist.set_yticks([])
ax_hist.set_xticklabels(["-1", "0"], fontsize=18)
#ax_hist.set_yticklabels(["0.5", "1"], fontsize=16)
ax_hist.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, pad=1)
ax_hist.xaxis.labelpad = -1.0
#ax_hist.set_xlim(left=-4, right=0)
#ax_hist.minorticks_on()
#ax_hist.grid(True, which='both', axis='y', ls='--', lw=0.2)
#for spine in ax_hist.spines.values():
#    spine.set_linewidth(1.1)
[ax_hist.spines[s].set_visible(False) for s in ax_hist.spines]
#ax_hist.set_xscale("log")
ax_hist.set_facecolor("0.9"); ax_hist.patch.set_alpha(1); ax_hist.set_zorder(0)


ax.set_xlim(left=0)
ax.set_ylim(bottom=bottom_power_limit, top=9.)
ax.set_xlabel(r'z (m)', fontsize=28)
ax.set_ylabel(r'$\mathrm{log_{10}P}$ (W)' , fontsize=28, labelpad=20)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.set_xticks([2.5, 5, 7.5])
ax.set_yticks([4, 6, 8])
#ax.set_xticklabels(["4", "8", "12"], fontsize=25)
ax.set_yticklabels(["4", "6", "8"], fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
#ax.minorticks_on()
#ax.tick_params(axis="both", which="minor", labelsize=14, length=3, width=0.9, direction="out")
ax_hist.set_facecolor("0.9"); ax_hist.patch.set_alpha(1); ax_hist.set_zorder(0)

ax.text(0.43, 0.29, r"$\mathbf{e/e^{\!+}}$", fontfamily="sans-serif", fontsize=20, color="crimson", transform=ax.transAxes)
ax.text(0.59, 0.39, r"e", fontfamily="sans-serif", fontsize=20, fontweight="bold", color="crimson", transform=ax.transAxes)



ax.text(0.05, 0.90, r"b", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)

######################################       RaDiO      ########################################################################################################################

labels = [r"$e^- \, / \,\, e^+$ (ellipsoidal)", r"$e^-$ (ellipsoidal)"]

ax = fig.add_subplot(gs[0, 2])
ax_RaDiO_0 = ax
#ax.set_box_aspect(1)
fill_color = [True, True]
linewidths = [0.75, 1.25]


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
        window_start = int(len(signal_time) * 0.015) # 0
        window_end = int(len(signal_time) * 0.70) # len(signal_time)
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
        kernel_size = 25
        kernel = np.ones(kernel_size) / kernel_size
        power_spectral_density[0:HF_limit] = np.convolve(power_spectral_density[0:HF_limit], kernel, mode='same')
        ax.plot(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0],
                 power_spectral_density[0:HF_limit],
                 c=colors[simulation_results_path_index], linewidth=linewidths[simulation_results_path_index],  label=labels[simulation_results_path_index]) # Normalized PSD with HF-truncated
        if fill_color[simulation_results_path_index]:
            ax.fill_between(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0],
                             power_spectral_density[0:HF_limit], 0,
                             color=colors[simulation_results_path_index],
                             alpha=.5)
        np.set_printoptions(threshold=np.inf)
        print(simulation_results_path)
        collected_spectra.append(power_spectral_density[0:HF_limit])
        freqs = frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0]
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


main_freq = freqs[np.argmax(np.array(collected_spectra[1]))]
print("w_0 (in eV) --> ", main_freq)
ax.set_xlabel(r"eV")
ax.set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=28)
ax.set_xlim(0.7 * main_freq, 7.5*main_freq)
ax.set_ylim(1e13, 1e18)
ax.set_xscale("linear")
ax.set_yscale("log")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.legend(fontsize="xx-large", loc='upper right', frameon=True, markerscale=120)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider  = make_axes_locatable(ax)
ax_ratio = divider.append_axes("bottom", size="30%", pad=0.12, sharex=ax)
plt.setp(ax.get_xticklabels(), visible=False)  # hide top panel x labels
ax_ratio.spines['top'].set_visible(False)
ax_ratio.set_ylabel("ratio", fontsize=28)
ax_ratio.axhline(1.0, lw=0.8, alpha=0.6)
ax_ratio.set_xscale(ax.get_xscale())

# freq: (N,), amp: (N,), targets: (M,)
rms_freq = 0.05
R = np.ones(len(collected_spectra[0]))
targets = np.array([main_freq, 3*main_freq, 5*main_freq, 7*main_freq])
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

ax_ratio.plot(collected_freqs[0], R, lw=3, color="goldenrod")
ax_ratio.set_ylim(bottom=1e-1, top=1e1)
ax_ratio.set_yscale("log")
ax_ratio.set_xlabel(r"eV", fontsize=28)  # reuse label if you already set i


ax_ratio.set_xticks([30, 60, 90, 120])
ax_ratio.set_yticks([1e-1, 1, 1e1])
ax_ratio.set_xticklabels(["30", "60", "90", "120"], fontsize=25)
ax_ratio.set_yticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"], fontsize=25)
ax.set_yticks([1e14, 1e15, 1e16, 1e17])
ax.set_yticklabels([r"$10^{14}$", r"$10^{15}$", r"$10^{16}$", r"$10^{17}$"], fontsize=25)
ax_ratio.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
ax.tick_params(axis="y", which="major", labelsize=25, length=8, width=1.5, direction="out")


ax.text(0.06, 0.9, r"n = 1", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.37, 0.62, r"n = 3", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.67, 0.38, r"n = 5", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.83, 0.24, r"n = 7", fontfamily="sans-serif", fontsize=18, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)

ax.text(0.15, 0.62, r"c", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


###############################################################################################################################################################################


simulation_results_paths = ['/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_UV_ep_pancake_3/simulation_results/',
                            '/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/draft_reference_UV_e_ellipsoidal/simulation_results/',
                            ]

labels = [r"$e^- \, / \,\, e^+$ (pancake)", r"$e^-$ (ellipsoidal)"]
colors = ["orangered", "mediumblue"]


right_col = gs[:, 4].subgridspec(3, 1, height_ratios=[0.5, 2., 0.5], hspace=0)
ax = fig.add_subplot(right_col[1, 0])
#ax.set_box_aspect(1)

fill_color = [True, True]
linewidths = [1.0, 2.0]

collected_spectra = []
freqs = []
post_saturation_end_times = [0.80, 0.90]
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
        window_start = int(len(signal_time) * 0.015) # 0
        window_end = int(len(signal_time) * post_saturation_end_times[simulation_results_path_index]) # len(signal_time)
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
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        power_spectral_density[0:HF_limit] = np.convolve(power_spectral_density[0:HF_limit], kernel, mode='same')
        np.set_printoptions(threshold=np.inf)
        print(simulation_results_path)
        collected_spectra.append(power_spectral_density[0:HF_limit])
        freqs.append(frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0])
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

# Exact alignment
main_freq_0 = freqs[0][np.argmax(np.array(collected_spectra[0]))]
main_freq_1 = freqs[1][np.argmax(np.array(collected_spectra[1]))]
if main_freq_1 > main_freq_0:
    freqs[1] -= main_freq_1 - main_freq_0
else:
    freqs[0] -= main_freq_0 - main_freq_1
for simulation_results_path_index, simulation_results_path in enumerate(simulation_results_paths):
    ax.plot(freqs[simulation_results_path_index],
             collected_spectra[simulation_results_path_index],
             c=colors[simulation_results_path_index], linewidth=linewidths[simulation_results_path_index],  label=labels[simulation_results_path_index]) # Normalized PSD with HF-truncated
    if fill_color[simulation_results_path_index]:
        ax.fill_between(freqs[simulation_results_path_index],
                         collected_spectra[simulation_results_path_index], 0,
                         color=colors[simulation_results_path_index],
                         alpha=.5)

main_freq = freqs[0][np.argmax(np.array(collected_spectra[0]))]
print("w_0 (in eV) --> ", main_freq)
ax.set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=28)
ax.set_xlim(0.7 * main_freq, 7.5*main_freq)
ax.set_ylim(5e13, 1.25e20)
ax.set_xscale("linear")
ax.set_yscale("log")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.legend(fontsize="xx-large", loc='upper right', frameon=True, markerscale=120)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider  = make_axes_locatable(ax)
ax_ratio = divider.append_axes("bottom", size="25%", pad=0.12, sharex=ax)
plt.setp(ax.get_xticklabels(), visible=False)  # hide top panel x labels
ax_ratio.spines['top'].set_visible(False)
ax_ratio.set_ylabel("ratio", fontsize=28, labelpad=18)
ax_ratio.axhline(1.0, lw=0.8, alpha=0.6)
ax_ratio.set_xscale(ax.get_xscale())

# freq: (N,), amp: (N,), targets: (M,)
rms_freq = 0.05
R = np.ones(len(freqs[0]))
targets = np.array([main_freq, 3*main_freq, 5*main_freq, 7*main_freq])
idx_0 = np.array([
    (i := np.where((freqs[0] >= t * (1 - rms_freq) ) & (freqs[0] <= t * (1 + rms_freq)))[0])[np.argmax(collected_spectra[0][i])] ###   Focus only for odd harmonics, suppress others
    for t in targets
])
val_0 = collected_spectra[0][idx_0]
idx_1 = np.array([
    (i := np.where((freqs[1] >= t * (1 - rms_freq) ) & (freqs[1] <= t * (1 + rms_freq)))[0])[np.argmax(collected_spectra[1][i])] ###   Focus only for odd harmonics, suppress others
    for t in targets
])
val_1 = collected_spectra[1][idx_1]
R[idx_0] = val_0 / val_1
print(val_0, val_1, val_0 / val_1)

ax_ratio.plot(freqs[0], R, lw=3, color="goldenrod")
ax_ratio.set_ylim(bottom=1, top=3e2)
ax_ratio.set_yscale("log")
ax_ratio.set_xlabel(r"eV", fontsize=28)  # reuse label if you already set i


ax_ratio.set_xticks([30, 60, 90, 120])
ax_ratio.set_yticks([1e0, 1e1, 1e2])
ax_ratio.set_xticklabels(["30", "60", "90", "120"], fontsize=25)
ax_ratio.set_yticklabels([r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"], fontsize=25)
ax.set_yticks([1e15, 1e17, 1e19])
ax.set_yticklabels([r"$10^{15}$", r"$10^{17}$", r"$10^{19}$"], fontsize=25)
ax_ratio.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
ax.tick_params(axis="y", which="major", labelsize=25, length=8, width=1.5, direction="out")


ax.text(0.07, 0.92, r"n = 1", fontfamily="sans-serif", fontsize=20, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.36, 0.67, r"n = 3", fontfamily="sans-serif", fontsize=20, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.65, 0.50, r"n = 5", fontfamily="sans-serif", fontsize=20, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)
ax.text(0.83, 0.40, r"n = 7", fontfamily="sans-serif", fontsize=20, fontstyle="italic", fontweight="semibold", transform=ax.transAxes)


################   Artists  #################

con_A = ConnectionPatch(
    xyA=(1.0, 0.99), coordsA="axes fraction",
    xyB=(0, 1), coordsB="axes fraction",
    axesA=ax_RaDiO_0, axesB=ax,
    linestyle="--", linewidth=2, zorder=4, clip_on=False
)
fig.add_artist(con_A)

con_B = ConnectionPatch(
    xyA=(1.0, 0.), coordsA="axes fraction",
    xyB=(0., 0.), coordsB="axes fraction",
    axesA=ax_RaDiO_1, axesB=ax_ratio,
    linestyle="--", linewidth=2, zorder=4, clip_on=False
)
fig.add_artist(con_B)


ax.text(0.15, 0.60, r"g", fontfamily="sans-serif", fontsize=28, fontweight="bold", transform=ax.transAxes)


###############################################################################################################################################################################
#fig.tight_layout()
pathlib.Path(figname + ".png").unlink(missing_ok=True)
fig.savefig(figname + ".png", format="png", dpi=300)
plt.close(fig)


