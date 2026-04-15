import logging
import pathlib
import mpmath as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from numpy import ndarray


def setup_logger(name="app", enabled=True, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()      # you control handlers here
    logger.propagate = False

    if enabled:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(level)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)

    return logger


#########################################################################################################################################################

# ==============================
# Xie 3d correction factors over 1d pierce parameter and beam parameters
# ==============================
def xie_3d_from_rho(
    lambda_u: float,
    lambda_r: float,
    rho_1d: float,
    sigma_r: float,          # rms transverse size (round beam)
    epsn: float,             # normalized emittance [m·rad]
    gamma: float,
    sigma_eta: float,        # rms relative energy spread (slice, if you prefer)
    beta_av: float | None = None
):
    """
    M. Xie 3D correction: Lg = Lg0(1+Λ), with Λ(Xd, Xε, Xγ).
    Returns (Lambda_xie, Lg0, Lg3d, rho_eff, diagnostics_dict).
    Reference: Eqs. (6.25)–(6.27) in FEL book.
    """
    # 1D power gain length (Eq. 6.25 uses Eq. 4.53 definition)
    Lg0 = lambda_u / (4 * np.pi * np.sqrt(3) * rho_1d)
    # geometric emittance
    eps = epsn / gamma
    # if no optics provided, infer a matched beta from sigma^2 = beta * eps
    if beta_av is None:
        beta_av = sigma_r**2 / eps
    # dimensionless Xie parameters (defined around Eq. 6.21; used in Eq. 6.26)
    Xd = Lg0 * lambda_r / (4 * np.pi * sigma_r**2)
    Xeps = Lg0 * 4 * np.pi * eps / (beta_av * lambda_r)
    Xg = Lg0 * 4 * np.pi * sigma_eta / lambda_u

    # fitted coefficients (Eq. 6.27)
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19 = (
        0.45,0.57,0.55,1.6,3.0,2.0,0.35,2.9,2.4,51,0.95,3.0,5.4,0.7,1.9,1140,2.2,2.9,3.2
    )
    # Λ parametrization (Eq. 6.26)
    Lambda_xie = (
        a1*Xd**a2
        + a3*Xeps**a4
        + a5*Xg**a6
        + a7*Xeps**a8 * Xg**a9
        + a10*Xd**a11 * Xg**a12
        + a13*Xd**a14 * Xeps**a15
        + a16*Xd**a17 * Xeps**a18 * Xg**a19
    )

    # 3D power gain length (Eq. 6.25)
    Lg3d = Lg0 * (1 + Lambda_xie)
    # "effective rho" consistent with the *definition* Lg ≡ λu/(4π√3 ρ_eff)
    rho_eff = rho_1d / (1 + Lambda_xie)

    diag = dict(Lg0=Lg0, Lg3d=Lg3d, Xd=Xd, Xeps=Xeps, Xg=Xg, beta_av=beta_av, eps=eps)
    return Lambda_xie, Lg0, Lg3d, rho_eff, diag


# ==============================
# FEL dispersion relation
# Λ^3 + 2 i s Λ^2 + (β^2 - s^2) Λ - i = 0
# with:
#   Λ    = alpha / Gamma
#   s    = eta / rho
#   β    = k_p / Gamma
# ==============================
def fel_eigenvalues(eta, rho, Gamma, kp=0.0):
    """
    Solve the 1D high-gain FEL dispersion relation (Schmüser-style):

        Λ^3 + 2 i s Λ^2 + (β^2 - s**2) Λ - i = 0

    with:
        Λ    = alpha / Gamma
        s    = eta / rho
        β    = kp / Gamma

    Inputs:
        eta   : relative energy detuning (W - W_r)/W_r (dimensionless)
        rho   : FEL parameter (dimensionless)
        Gamma : gain parameter [1/m]
        kp    : longitudinal space charge parameter [1/m]

    Returns:
        alpha_j [1/m] : 3 complex eigenvalues.
    """
    s = eta / rho
    beta = kp / Gamma

    # Coefficients of cubic in Λ
    # Λ^3 + 2 i s Λ^2 + (β^2 - s^2) Λ - i = 0
    coeffs = [1.0,
              2j * s,
              (beta**2 - s**2),
              -1j]

    roots_Lambda = np.roots(coeffs)
    return roots_Lambda * Gamma


# ---------- 1D Pierce parameter (Wolski-like, axisymmetric beam) ---------- #
def rho_1d(I, sigma_x, sigma_y, gamma, lam_u, K, JJ):
    # Dimensionless, includes current density via sigma_x*sigma_y
    num = (I / I_A) * (K**2 * JJ**2) * lam_u**2
    den =  64 * math.pi**2 * gamma**3 * sigma_x * sigma_y
    return (num / den)**(1.0 / 3.0)


# ---------- longitudinal electric field calculator ---------- #
def Ez_onaxis(z, L, R, rho, eps0, gamma=1.0):
    """
    On-axis longitudinal electric field of a uniformly charged finite cylinder,
    centered at z=0 (bunch from -L/2 to +L/2), valid for all z on axis.
    """
    z = np.array(z)
    term_sqrt = np.sqrt(R**2 + gamma**2*(L/2 - z)**2) - np.sqrt(R**2 + gamma**2*(L/2 + z)**2)
    term_abs = np.abs(z + L/2) - np.abs(z - L/2)
    return (rho / (2 * eps0 * gamma)) * (term_sqrt + gamma * term_abs)


# ---------- Analytic field slope at z=0 from Taylor expansion (first derivative) ---------- #
def G_field_analytic(L, R, rho, eps0, gamma=1.0):
    f0 = np.sqrt(R**2 + gamma**2 * (L**2) / 4.0)
    return (rho / eps0) * (1.0 - (gamma * L) / (2.0 * f0))


# ---------- Geometry factor approximation F(x) ---------- #
def F_geom(x):
    # Smooth interpolant with correct limits:
    # F ~ x^2 for x << 1 (suppressed Ez), F ~ 1 for x >> 1
    return x**2 / (1.0 + x**2)


#########################################################################################################################################################
### Physical constants
e = 1.602176634e-19
m_e = 9.10938356e-31
c = 2.99792458e8
eps0 = 8.854187817e-12
Z0 = 376.730313668
mu0 = 4.0 * np.pi * 1e-7  # H/m
I_A = 17045.0  # Alfven current [A]


#########################################################################################################################################################

######################  FEL Gain with Space Charge Metric ###########################

mesh_vis = True
instance_EField_vis = False
print_info = False
log = setup_logger(enabled=print_info)

# Fixed beam / undulator parameters (For HEATMAP Visualization)
###
E_GeV = 10   # in GeV
gamma = E_GeV * 1e9 / (0.511e6)  # relativistic factor
R = 250e-6                        # fixed beam radius [m]
normalized_emittance = 3e-1        # um-rad (WEAK FOCUSING, if you check the corresponding beta function)
bunch_energy_spread = 1e-4       # initial energy spread
lam_u = 0.03                     # undulator period [m]
K = 1.5                           # undulator parameter
ku = 2 * math.pi / lam_u
# Scan Ranges
Q_vals = np.logspace(-10.4, -8.5, 100)#np.array([50e-12])      # total charge: in Coulombs
Lb_vals = np.logspace(-7, -5, 100)#np.array([10e-6])    # bunch length: in meters
###

e_p_fraction = 0    # positive / minus charge fraction (default is 0, corresponds to electron-only bunch)

######################

sigma_x = R / 2.35
sigma_y = R / 2.35
rho_grid = np.zeros((len(Q_vals), len(Lb_vals)))
P_sat3d_grid = np.zeros_like(rho_grid)
DC_grid = np.zeros_like(rho_grid) # scaled DC detuning (eta_coherent / rho)
Xi_grid = np.zeros_like(rho_grid)

for i, Q in enumerate(Q_vals):
    for j, Lb in enumerate(Lb_vals):
        log.info("%d, %d", i, j)

        # Particle e.o.m. stats
        transverse_oscillation_amplitude = K / (gamma * ku) * 1e6   # in um.
        log.info("transverse oscillation amplitude (in um): %.2f", transverse_oscillation_amplitude)

        # JJ factor for planar undulator
        xi = K**2 / (4 + 2 * K**2)
        JJ = float(mp.besselj(0, xi) - mp.besselj(1, xi))

        # Peak current
        I = Q * c / Lb
        if e_p_fraction != 0:
            I *= (0.5 + e_p_fraction / 2)
        log.info("Beam Current (kA) -> %.2f", I / 1e3)

        # Radiation wavelength
        Lambda_r = lam_u / (2 * gamma**2) * (1 + K**2 / 2)
        log.info("Radiation wv. (nm) & # of expected microbunches --> %.3f  &  %d", Lambda_r * 1e9, int(Lb / Lambda_r) )
        log.info("1st harmonic energy (eV) --> %.2f",  1239.841984 / (Lambda_r * 1e9))

        # 1D Pierce parameter
        rho_1d_val = rho_1d(I, sigma_x, sigma_y, gamma, lam_u, K, JJ)

        # --- Xie 3D correction (power gain length) ---
        sigma_r = np.sqrt(sigma_x * sigma_y)             # rms size (round-beam proxy)
        epsn = normalized_emittance * 1e-6               # [μm·rad] -> [m·rad]
        sigma_eta = bunch_energy_spread / 1              # keep your slice proxy (1 / 3) or projected bunch spread ( / 1 )
        Lambda_xie, Lg0, Lg3d, rho_eff, xdiag = xie_3d_from_rho(
            lambda_u=lam_u,
            lambda_r=Lambda_r,
            rho_1d=rho_1d_val,
            sigma_r=sigma_r,
            epsn=epsn,
            gamma=gamma,
            sigma_eta=sigma_eta,
            beta_av=None                                # or set a fixed beta here if you prefer
        )
        log.info("Xie 3D correction --> Lambda, gain lengths (1d - 3d), pierce parameters (1d - 3d)\n %.3f  --  %.3f  --  %.3f  --  %.5f  --  %.5f",
                 Lambda_xie, Lg0, Lg3d, rho_1d_val, rho_eff)
        log.info("Xie correction parameters --> %.3f  --  %.3f  --  %.3f", xdiag["Xd"], xdiag["Xeps"], xdiag["Xg"])
        log.info("Matched beta (in meters) --> %.1f", xdiag["beta_av"])

        # Use rho_eff for the eigenmode scaling (heuristic but definition-consistent)
        rho = rho_eff    ### otherwise, 1d FEL value --> rho_1d_val * Correction_Factor
        rho_grid[i, j] = rho

        # Beam power and 3D-corrected saturation power
        P_beam = gamma * m_e * c**2 * (I / e)
        P_sat3d_grid[i, j] = 1.6 * rho_eff * P_beam # 1.6 * rho_1d_val * P_beam * (Lg0 / Lg3d)**2
        log.info("Saturation power (GW) --> %.2f", P_sat3d_grid[i, j] / 1e9)

        if rho <= 0:
            Xi_grid[i, j] = np.nan
            continue

        # 3D gain length and interaction length (~saturation length)
        Lg = Lg3d ###  lam_u / (4 * math.pi * math.sqrt(3) * rho)
        log.info("Gain length (m) --> %.2f", Lg)
        L_int = 18.0 * Lg
        log.info("Saturation length (m) --> %.2f", L_int)

        #### Longitudinal space-charge field from low-k net-charge mode
        kb = 2.0 * math.pi / Lb
        x = kb * R / gamma
        F = F_geom(x)
        # |Z_parallel|/L and effective Ez
        #E0 = Z0 * kb * I / (2.0 * math.pi * gamma**2) * F
        #E_eff = 0.65 * E0  # ~1 * std  * amplitude for sinusoidal-like profile

        #### Longitudinal space-charge field from exact electrostatic calculation
        sigma_r = sigma_x       # rms transverse size [m]
        A_b = np.pi * sigma_x**2  # effective beam cross-section [m^2]
        n_e = I / (e * c * A_b) # 1D electron density --> uniform transverse density (assuming v_z ~ c)
        n_lab = n_e * e  # volume charge density
        if e_p_fraction != 0:
            n_lab *= (1 - e_p_fraction) / 2 / (0.5 + e_p_fraction / 2)
        z_vals = np.linspace(-Lb/2, Lb/2, 1000)
        Ez_vals = Ez_onaxis(z_vals, Lb, R, n_lab, eps0, gamma)
        trim = int(len(Ez_vals))
        left_trim_frac = 0.1
        right_trim_frac = 0.9
        E_0_electrostatic_model = np.mean(np.abs(Ez_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)]) )
        E_eff = E_0_electrostatic_model
        log.info("Calculated mean E_z (V/m) --> %.2f", E_eff)

        # Coherent energy detuning over L_int
        eta_coh = e * E_eff * L_int / (gamma * m_e * c**2)
        DC_grid[i, j] = eta_coh / rho

        # Plasma term k_p / Gamma
        n_e = Q / (math.pi * R**2 * Lb * e)
        omega_p2 = n_e * e**2 / (eps0 * m_e)
        kp = math.sqrt(max(omega_p2 / (gamma**3 * c**2), 0.0))
        Gamma = 2.0 * ku * rho
        mu = kp / Gamma if Gamma > 0 else 0.0

        # Cumulative normalized LSC detuning parameter
        Xi = math.sqrt((eta_coh / rho)**2 + mu**2)
        Xi_grid[i, j] = Xi
        log.info("LSC ingredients (DC, AC, total) --> %.3f , %.3f, %.3f", eta_coh / rho, mu, Xi_grid[i, j])

        # eigenmodes
        eta_intrinsic_offset_RMS = bunch_energy_spread / 3  # bunch energy spread input is considered as an expected projected RMS spread value for a given slice
        eta_total = eta_intrinsic_offset_RMS + eta_coh
        roots = fel_eigenvalues(eta_total, rho, Gamma, kp=kp)
        a_with_LSC = roots[np.argmax(roots.real)]
        roots = fel_eigenvalues(eta_intrinsic_offset_RMS, rho, Gamma, 0)
        a_wo_LSC = roots[np.argmax(roots.real)]
        log.info("Eigenvalue problem growing modes --> %.5f, %.5f, %.5f, %.5f, %.5f", rho, bunch_energy_spread, eta_total, np.real(a_wo_LSC), np.real(a_with_LSC))

        # number of spikes from the cooperation_length / bunch_length
        cooperation_length = Lambda_r / (4*np.pi*rho)
        log.info("Coop. length (um), bunch length (um) --> %.6f, %.6f", cooperation_length*1e6,  Lb*1e6)
        log.info("Expected # of SASE spikes at saturation --> %d", max(1, int(np.rint(Lb / (2 * math.pi * cooperation_length)))) )

######################

if mesh_vis:
    # Prepare mesh for plotting
    Lb_mesh, Q_mesh = np.meshgrid(Lb_vals, Q_vals)

    # Convert saturation power to GW for readability
    P_sat3d_GW = P_sat3d_grid / 1e9

    # Create figure
    fig, ax = plt.subplots(figsize=(6,5))

    # Heatmap of 3D saturation power [GW]
    hm = ax.pcolormesh(Lb_mesh*1e6, Q_mesh*1e12, P_sat3d_GW,
                       cmap='cividis', norm=LogNorm(vmin=2, vmax=5e4), shading='auto')

    # Axis
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$L_{\\mathrm{beam}} (\\mathrm{\mu m})$', fontsize=17)
    ax.set_ylabel('$Q_{\\mathrm{beam}}$ (pC)', fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=17, length=8, width=1.5, direction="out")

    cbar = fig.colorbar(hm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('$P_{\\mathrm{sat}}$ (GW)', fontsize=17)
    cbar.set_ticks(np.logspace(1, 4, 4))
    cbar.set_ticklabels([f'$10^{{{i}}}$' for i in range(1, 5)])
    cbar.ax.tick_params(length=7, width=1.2, labelsize=17)

    # Contours of combined LSC detuning parameter Xi
    levels = [0.2, 0.7, 2.0]
    labels = [r'$\Lambda_{LSC}=$' + str(levels[0]), r'$\Lambda_{LSC}=$'  + str(levels[1]), r'$\Lambda_{LSC}=$' +  str(levels[2])]
    cs = ax.contour(Lb_mesh*1e6, Q_mesh*1e12, Xi_grid, levels=levels,
                    linewidths=2.5, colors = ['black', 'red', 'dodgerblue'], linestyles="--")
    handles = []
    for level, coll in zip(cs.levels, cs.collections):
        # For LineCollection, colors are in edgecolor or facecolor array
        ec = coll.get_edgecolor()
        color = ec[0] if len(ec) else 'k'
        handles.append(Line2D([0], [0],
                              color=color,
                              linestyle='--'))
        labels.append(f"Level {level:g}")

    ax.legend(handles, labels, loc=(0.01, 0.43))

    ax.grid(False)  # or: ax.grid(visible=False, which='both', axis='both')
    ax.minorticks_off()
    #ax.grid(True, which='both', ls='--', lw=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    ax.text(0.03, 0.89, r"$\mathrm{E_{b} \approx 10\,GeV}$", fontfamily="sans-serif", fontsize=13, fontweight="roman", transform=ax.transAxes)
    ax.text(0.03, 0.82, r"$\mathrm{R_{b} \approx 250\, \mu m}$", fontfamily="sans-serif", fontsize=13, fontweight="roman", transform=ax.transAxes)
    ax.text(0.03, 0.75, r"$\mathrm{K=1.5}$", fontfamily="sans-serif", fontsize=13, fontweight="roman", transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

    pathlib.Path("LSC_Detuning" + ".png").unlink(missing_ok=True)
    fig.savefig("LSC_Detuning" + ".png", format="png", dpi=200)
    plt.close(fig)

######################

######################  Bunch Electrostatic field characteristics check ###########################

if instance_EField_vis:
    L_lab = Lb_vals[0]
    I_peak = Q_vals[0] / (L_lab / c) # peak current [A]
    sigma_r = sigma_x       # rms transverse size [m]
    A_b = np.pi * sigma_x**2  # effective beam cross-section [m^2]
    n_e = I_peak / (e * c * A_b) # 1D electron density --> uniform transverse density (assuming v_z ~ c)
    n_lab = n_e * e  # volume charge density

    G_ana_labframe = G_field_analytic(L_lab, R, n_lab, eps0, gamma)
    # Numerical slope via central difference
    dz = L_lab * 1e-5
    Ez_p = Ez_onaxis(dz, L_lab, R, n_lab, eps0, gamma)
    Ez_m = Ez_onaxis(-dz, L_lab, R, n_lab, eps0, gamma)
    G_num = (Ez_p - Ez_m) / (2.0 * dz)
    # Numerical second derivative at 0 (should be ~0 for odd Ez)
    Ez_0 = Ez_onaxis(0.0, L_lab, R, n_lab, eps0, gamma)
    Ez_pp = (Ez_p - 2.0 * Ez_0 + Ez_m) / (dz**2)
    print("Analytic slope G_field [V/m^2]:", G_ana_labframe)
    print("Numeric  slope G_field [V/m^2]:", G_num)
    print("Relative difference:", (G_num - G_ana_labframe) / G_ana_labframe)
    print("Second derivative at 0 (should be ~0):", Ez_pp)
    # Check the invariance of the longitudinal electric field
    G_ana_bunchrestframe = G_field_analytic(L_lab * gamma, R, n_lab / gamma, eps0, gamma=1)
    print("Linearized longitudinal fields in lab frame and bunch rest frame [V/m]:", G_ana_labframe * L_lab, G_ana_bunchrestframe * L_lab * gamma)

    # Now compare actual Ez vs linearized field G_field * z
    z_vals = np.linspace(-L_lab/2, L_lab/2, 1000)
    Ez_vals = Ez_onaxis(z_vals, L_lab, R, n_lab, eps0, gamma)
    Ez_lin = G_ana_labframe * z_vals
    trim = int(len(Ez_vals))
    left_trim_frac = 0.1
    right_trim_frac = 0.9
    plt.figure(figsize=(6,4))
    plt.plot(z_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)], Ez_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)], label="Exact $E_z(z)$")
    plt.plot(z_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)], Ez_lin[int(left_trim_frac * trim) : int(right_trim_frac * trim)], linestyle="--", label=r"Linearized $G_{\mathrm{field}} z$")
    plt.plot(z_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)], z_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)] * 0, "-")
    #print(np.std(np.abs(Ez_lin)), np.mean(np.abs(Ez_lin)))
    #print(np.std(np.abs(Ez_vals)), np.mean(np.abs(Ez_vals)))
    plt.xlabel("z (m) [centered at bunch]")
    plt.ylabel(r"$E_z$ (V/m)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    E_0_electrostatic_model: ndarray = np.mean(np.abs(Ez_vals[int(left_trim_frac * trim) : int(right_trim_frac * trim)]) )
    #print(E_0_electrostatic_model)



#########################################################################################################################################################
'''
# ----- Fixed beam / undulator parameters (For HEATMAP Visualization)
########################################################################################################################
E_GeV = 48   # in GeV
gamma = E_GeV * 1e9 / (0.511e6)  # relativistic factor
R = 15e-6                        # fixed beam radius [m]
normalized_emittance = 3e-1        # um-rad (WEAK FOCUSING, if you check the corresponding beta function)
bunch_energy_spread = 1e-4       # initial energy spread
lam_u = 0.03                     # undulator period [m]
K = 2.5                           # undulator parameter
ku = 2 * math.pi / lam_u
# Scan Ranges
Q_vals = np.array([5e-12])#np.logspace(-10.4, -8.5, 100)#np.array([50e-12])      # total charge: in Coulombs
Lb_vals = np.array([4e-9])#np.logspace(-7, -5, 100)#np.array([10e-6])    # bunch length: in meters
########################################################################################################################
E_GeV = 0.35   # in GeV
gamma = E_GeV * 1e9 / (0.511e6)  # relativistic factor
R = 450e-6                        # fixed beam radius [m]
normalized_emittance = 3e-1        # um-rad (WEAK FOCUSING, if you check the corresponding beta function)
bunch_energy_spread = 1e-3       # initial energy spread
lam_u = 0.031                     # undulator period [m]
K = 1.5                          # undulator parameter
ku = 2 * math.pi / lam_u
# Scan Ranges
Q_vals = np.array([400e-12])#np.logspace(-10.4, -8.5, 100)#np.array([50e-12])      # total charge: in Coulombs
Lb_vals = np.array([10e-6])#np.logspace(-7, -5, 100)#np.array([10e-6])    # bunch length: in meters
'''
