from __future__ import annotations
import sys, os, logging, traceback, pathlib
import glob
import math
import numpy as np
from numpy import ndarray
import scipy.constants
from scipy.constants import epsilon_0, mu_0, h, c, e, m_e, electron_volt
from scipy.constants import physical_constants
import scipy.integrate as integrate
from scipy.special import jv, erfcx
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib import colors
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial
from numbers import Real

figname="HardXray"
#########################################################################################################################################################
""" Physical constants """
I_A = 4.0 * np.pi * epsilon_0 * m_e * c**3 / e  # Alfven current in Amperes [17045.09]
#########################################################################################################################################################


### RaDiO class
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


##### Setup logger
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

### Initial log
log = setup_logger()
log.info("Python interpreter: " + sys.executable)
ANALYSIS_DIR_NAME=os.path.dirname(__file__)
log.info("Directory of the currently executed python script: " + ANALYSIS_DIR_NAME)


## Import happi module
try:
    import happi
    log.info("happi module is accessed via direct import from the python script" )
except Exception as error:
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
            log.info("Diagnostics script path for the postprecess: " + check_file)
            process_info = script_key
            break
        except Exception as error:
            continue
            #logging.error(traceback.format_exc())
    if process_info is None:
        log.error("Diagnostics script could not be found, exit.")
        exit()

## Diagnostics units
unit_dictionary = {"dimensions": ["3d_charge_density", "weight", "angle", "position", "time", "velocity", "current", "energy", "integrated energy", "number density", "average charge", "electric field strength", "magnetic field strength", "Poynting vector"],
                   "units": ["e/cm^3", "m^-1", "degree", "um", "ps", "m/s", "A", "MeV", "joule/um", "m^-1", "C", "V/m", "T", "W/m^2"]
                   }



#########################################################################################################################################################
"""Exact free-space on-axis longitudinal self-field of a rigid round bunch.

All inputs use SI: radius [m], signed total_charge [C], full length [m]. The output is E_z [V/m].

The bunch is uniform along z in [-length/2, length/2].  Transversely:
  gaussian_profile=True : f(r) = exp(-r**2/(2*sigma**2))/(2*pi*sigma**2),
                          radius = sigma_x = sigma_y, NOT rms radial size.
  gaussian_profile=False: f(r) = 1/(pi*R**2) for r <= R, radius = R.

The source translates at a constant axial velocity with Lorentz factor gamma.
There are no conducting walls, acceleration/radiation fields, or evolution of
its prescribed density.  The transverse distribution is integrated exactly;
this is an on-axis reduction of the 3D Coulomb problem, not a line-charge model.

For a flat-top longitudinal profile the integrated Green function can be
applied to the whole occupied interval analytically.  No longitudinal source
mesh, FFT, image bunches, finite-difference gradient, or empirical gamma factor
is needed.  The returned sampling grid only samples the exact field.

References:
  B. Li, NIM A 1055 (2023) 168541, doi:10.1016/j.nima.2023.168541,
    Eqs. (3)-(5), (11)-(13), (19);
  V. Fusco, Beam Dynamics and Collective Effects in the SPARC Project,
    Sec. 2.2, Eq. (2.5), https://inspirehep.net/files/ad988c605c5c7546cef0127f82d09dea
"""


__all__ = ["E_z_onaxis", "E_z_onaxis_at", "profile_coordinates",
           "slice_green_function", "N_PROFILE_POINTS"]

N_PROFILE_POINTS = 4001  # 4*m+1 includes both bunch ends and the centre exactly.
_NORMALIZED_GRID = np.linspace(-1.0, 1.0, N_PROFILE_POINTS)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = (_GL_X + 1.0) / 2.0
_GL_W = _GL_W / 2.0
_SQRT_PI = np.sqrt(np.pi)


def _scalar(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _boolean(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False.")
    return bool(value)


def _parameters(radius, total_charge, length, gamma, gaussian_profile):
    radius = _scalar("radius", radius)
    total_charge = _scalar("total_charge", total_charge)
    length = _scalar("length", length)
    gamma = _scalar("gamma", gamma)
    gaussian_profile = _boolean("gaussian_profile", gaussian_profile)
    if radius <= 0 or length <= 0:
        raise ValueError("radius and length must be strictly positive.")
    if gamma < 1:
        raise ValueError("gamma must be >= 1.")
    aspect = gamma * (length / radius)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("gamma*length/radius is outside the supported float64 range.")
    return radius, total_charge, length, gamma, gaussian_profile, aspect


def profile_coordinates(length):
    """Coordinates [m] corresponding to E_z_onaxis: 4001 points on [-L, L]."""
    length = _scalar("length", length)
    if length <= 0:
        raise ValueError("length must be strictly positive.")
    return length * _NORMALIZED_GRID.copy()


def _gaussian_kernel_shape(x):
    """Stable 1 - sqrt(pi)*x*erfcx(x) for x >= 0.

    For large x this difference loses digits.  Ten asymptotic terms at x>=26
    have a truncation error far below double-precision roundoff.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    large = x >= 26.0
    xs = x[~large]
    result[~large] = 1.0 - _SQRT_PI * xs * erfcx(xs)
    t = 0.5 / x[large] / x[large]
    term = t.copy()
    series = term.copy()
    for n in range(1, 10):
        term *= -(2*n + 1) * t
        series += term
    result[large] = series
    return result


def E_z_onaxis_at(z, radius, total_charge, length, gamma, gaussian_profile=True):
    """Evaluate E_z [V/m] at arbitrary bunch-centred coordinates z [m].

    Parameters otherwise match E_z_onaxis.  The z argument can be a scalar
    or an array of any shape.  Use this helper for adaptive edge sampling or
    points farther away than the main function's fixed interval [-L, L].
    Positive z is toward the bunch head.  For Q>0, E_z>0 at z>0; use Q<0
    for an electron bunch.
    """
    radius, q, length, gamma, gauss, aspect = _parameters(
        radius, total_charge, length, gamma, gaussian_profile)
    z_array = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z_array)):
        raise ValueError("All z coordinates must be finite.")
    shape = z_array.shape
    z_flat = z_array.reshape(-1)
    if q == 0:
        out = np.zeros_like(z_flat)
    else:
        u = np.abs(z_flat) / length
        sign = np.sign(z_flat)
        clipped = np.minimum(2.0*u, 1.0)
        if not gauss:
            # Rationalized difference of sqrt(1+t^2)-t.  In particular,
            # b-a is evaluated from z directly, not by subtracting large
            # nearly equal distances.  This preserves near-centre fields.
            # Subtract dimensional coordinates BEFORE dividing. Near an
            # edge, z/L - 1/2 would lose the tiny distance to the boundary.
            a = (gamma / radius) * np.abs(np.abs(z_flat) - length/2)
            b = (gamma / radius) * (np.abs(z_flat) + length/2)
            ha, hb = np.hypot(1.0, a), np.hypot(1.0, b)
            ratio = 1.0 + (a + b) / (ha + hb)
            out = (q / (2.0*np.pi*epsilon_0*radius**2) * sign * clipped
                   * ratio / (ha + a) / (hb + b))
        else:
            scale = gamma / (np.sqrt(2.0) * radius)
            a = scale * np.abs(np.abs(z_flat) - length/2)
            b = scale * (np.abs(z_flat) + length/2)
            delta = scale * np.minimum(2*np.abs(z_flat), length)  # exact b-a
            base = q / (4.0*np.pi*epsilon_0*radius**2)
            out = np.empty_like(u)
            close = delta <= 1.0e-3 * (1.0 + a)
            far = ~close
            out[far] = (base * np.sqrt(np.pi/2.0) / aspect
                        * (erfcx(a[far]) - erfcx(b[far])))
            # Avoid cancellation of nearly equal erfcx values. Integrate
            # their derivative over the small interval using 8-point Gauss
            # quadrature. This is a floating-point evaluation safeguard,
            # not a longitudinal density-grid approximation.
            nodes = a[close, None] + delta[close, None] * _GL_X
            average = _gaussian_kernel_shape(nodes) @ _GL_W
            out[close] = base * clipped[close] * average
            out *= sign
    out = out.reshape(shape)
    return float(out) if out.ndim == 0 else out


def E_z_onaxis(radius, total_charge, length, gamma,
               gaussian_profile=True, visualize_profile=False):
    """Return the on-axis longitudinal electric-field array in V/m.

    radius : float [m]
        Gaussian: single-Cartesian-axis rms size sigma_x=sigma_y.
        Uniform: physical radius of a round disk (half its full diameter).
    total_charge : float [C]
        Signed bunch charge; negative for electrons, positive for positrons.
    length : float [m]
        Full bunch length in the frame where the field is requested.
        Bunch support is [-length/2, +length/2]
    gamma : float, >=1
        Axial bulk Lorentz factor relative to that observation frame.
    gaussian_profile : bool, default True
        Gaussian transverse density if True; uniform round disk otherwise.
    visualize_profile : bool, default False
        If True, plot E_z versus z in metres and mark the bunch boundaries.

    Returns
    -------
    numpy.ndarray, shape (4001,)
        Signed E_z in V/m, at np.linspace(-length, length, 4001).
        Coordinates are also available from profile_coordinates(length).

    The bunch's rest-frame counterpart uses the SAME radius and Q, length
    gamma*length, and gamma=1.  Its field equals this array at corresponding
    positions z_rest=gamma*z_lab.
    A fixed uniform output grid can under-resolve a visually narrow edge
    layer of width radius/gamma; the field at each sample is still evaluated
    from the exact integral. Use E_z_onaxis_at for a refined display grid.
    """
    visualize = _boolean("visualize_profile", visualize_profile)
    z = profile_coordinates(length)
    field = E_z_onaxis_at(z, radius, total_charge, length, gamma,
                         gaussian_profile=gaussian_profile)
    if visualize:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(z, field, lw=1.7, label=r"$E_z$ on axis")
        ax.axvspan(-length/2, length/2, color="0.75", alpha=0.25,
                   label="Bunch support")
        ax.axvline(-length/2, color="0.5", ls=":", lw=1)
        ax.axvline(length/2, color="0.5", ls=":", lw=1)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set(xlabel=r"Bunch-centred $z$ [m]", ylabel=r"$E_z$ [V/m]",
               title=("Gaussian" if gaussian_profile else "Uniform disk")
                     + " transverse density; longitudinal flat top")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        ax.grid(alpha=0.2)
        ax.legend()
        plt.show()
    return field


def slice_green_function(z, radius, gamma, gaussian_profile=True):
    """On-axis slice kernel w(z), in V/(C m), for E=int w(z-z')lambda(z') dz'.

    This helper is useful for verification or extensions to other longitudinal
    profiles. Its rapidly varying kernel should be integrated over source
    cells, rather than sampled naively on a coarse mesh. w(0)=0 is the
    symmetric principal-value convention; one-sided limits are nonzero.
    """
    radius, _, _, gamma, gauss, _ = _parameters(
        radius, 1.0, radius, gamma, gaussian_profile)
    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("All z coordinates must be finite.")
    t = gamma * (np.abs(z) / radius)
    if gauss:
        out = np.sign(z) / (4*np.pi*epsilon_0*radius**2) * _gaussian_kernel_shape(t/np.sqrt(2))
    else:
        h = np.hypot(1.0, t)
        out = np.sign(z) / (2*np.pi*epsilon_0*radius**2) / h / (h+t)
    return float(out) if out.ndim == 0 else out



#########################################################################################################################################################
""" Lineariezed eigenvalue problem solver for the FEL growth rate calculation
- plasma oscillation term and detuning are handled within the same scope.
- 3d Xie corrections added explicitly.
"""
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
    Returns (Lambda_xie, Lg0, Lg3d, rho_eff, rho_perp, diagnostics_dict).
    rho_eff includes energy spread; rho_perp excludes it.
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

    # Transverse-only correction (Xg=0) for the kinetic solver.
    Lambda_perp = a1*Xd**a2 + a3*Xeps**a4 + a13*Xd**a14*Xeps**a15
    rho_perp = rho_1d / (1 + Lambda_perp)

    diag = dict(Lg0=Lg0, Lg3d=Lg3d, Xd=Xd, Xeps=Xeps, Xg=Xg, beta_av=beta_av, eps=eps)
    return Lambda_xie, Lg0, Lg3d, rho_eff, rho_perp, diag


# ==============================
# FEL dispersion relation
# Λ^3 + 2 i s Λ^2 + (β^2 - s^2) Λ - i = 0
# with:
#   Λ    = alpha / Gamma
#   s    = eta / rho
#   β    = k_p / Gamma
# ==============================
def gaussian_fel_response(lam, s, sigma):
    """Gaussian response <(lam+i*p)^(-2)>; mean p=s, RMS width sigma."""
    a = complex(lam) + 1j*s
    if sigma == 0:
        return 1/a**2
    w = a/(np.sqrt(2.0)*sigma)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if abs(w) > 25.0 and w.real >= 0:
            # Avoid cancellation when approaching the cold limit.
            t = 1/(2*w*w)
            term = t
            series = term
            for n in range(1, 12):
                term *= -(2*n+1)*t
                series += term
            return series/sigma**2
        return (1-np.sqrt(np.pi)*w*erfcx(w))/sigma**2


def fel_eigenvalues(eta, rho, Gamma, kp=0.0, *, warm_beam=False, sigma_eta=0.0):
    """Return spatial amplitude eigenvalues alpha [1/m].

    eta: signed MEAN fractional detuning; sigma_eta: RMS fractional spread.
    Use rho=rho_perp and Gamma=2*ku*rho_perp for the effective transverse
    model; energy spread enters only the Gaussian response, not rho_perp.
    Cold/zero-spread calls return the original three cubic roots. Warm calls
    return resolved growing roots, sorted by real part, or an empty array.
    The resolution floor is Re(alpha/Gamma)>1e-8; the returned warm roots
    are not a complete kinetic spectrum. All energies share one radiation field.

    Seggebrock et al., PRSTAB 16, 070703 (2013), Eqs. (1)-(2),
    doi:10.1103/PhysRevSTAB.16.070703. The scalar transverse correction is
    an approximation to 3D effects, not a full 3D kinetic solution.
    """
    from scipy.optimize import root

    eta, rho, Gamma, kp, sigma_eta = (
        _scalar(name, value) for name, value in
        [('eta', eta), ('rho', rho), ('Gamma', Gamma),
         ('kp', kp), ('sigma_eta', sigma_eta)])
    warm_beam = _boolean('warm_beam', warm_beam)
    if rho <= 0 or Gamma <= 0 or sigma_eta < 0:
        raise ValueError('rho and Gamma must be positive; sigma_eta >= 0.')
    s, mu, sigma = eta/rho, kp/Gamma, sigma_eta/rho
    if not np.all(np.isfinite([s, mu, sigma])):
        raise ValueError('Normalized parameters must be finite.')
    cold = np.roots([1.0, 2j*s, mu**2-s**2, -1j])
    if not warm_beam or sigma == 0:
        return cold*Gamma

    # lambda = (i - mu^2*lambda) * Gaussian response.
    def residual(lam):
        with np.errstate(all='ignore'):
            return lam-(1j-mu**2*lam)*gaussian_fel_response(lam,s,sigma)

    def fun(v):
        f = residual(complex(v[0],v[1]))
        return [f.real, f.imag]

    guesses = [complex(max(.03,z.real),z.imag) for z in cold]
    scale = 1/(1+sigma**2+mu**2)
    guesses += [complex(x,y) for x in [.03, .2, .8, scale]
                for y in [0., .5, -s, -s+sigma, -s-sigma]]
    found = []
    best_residual = np.inf

    def try_seed(guess):
        nonlocal best_residual
        result = root(fun,[guess.real,guess.imag],tol=2e-11)
        z = complex(*result.x)
        r = abs(residual(z))
        if np.isfinite(r):
            best_residual = min(best_residual,r)
        if (np.isfinite(z) and z.real > 1e-8 and
                r < 2e-9*max(1,abs(z)) and
                not any(abs(z-old)<1e-6*max(1,abs(z)) for old in found)):
            found.append(z)

    for guess in guesses:
        try_seed(guess)
    if not found:
        # Retry more broadly before reporting no resolved growing mode.
        span = max(2.,abs(mu)+3*sigma)
        for y in np.linspace(min(0,-s)-span,max(0,-s)+span,41):
            for x in [.005,.15,.7,scale]:
                try_seed(complex(x,y))
    if not found and best_residual > 1e-6:
        raise RuntimeError('Gaussian FEL root solver did not converge; inspect parameters.')
    found.sort(key=lambda z:z.real,reverse=True)
    return np.asarray(found,dtype=complex)*Gamma


# ---------- 1D Pierce parameter (Wolski-like, axisymmetric beam) ---------- #
def rho_1d(I, sigma_x, sigma_y, gamma, lam_u, K, JJ):
    # Dimensionless, includes current density via sigma_x*sigma_y
    num = (I / I_A) * (K**2 * JJ**2) * lam_u**2
    den =  64 * math.pi**2 * gamma**3 * sigma_x * sigma_y
    return (num / den)**(1.0 / 3.0)



#########################################################################################################################################################

######################  FEL Gain with Space Charge Metric ###########################

# Fixed beam / undulator parameters (For HEATMAP Visualization)
warm_beam = True  # False: cold cubic; True: Gaussian slice spread is considered in kinetic dispersion relation
###
E_GeV = 17   # in GeV
gamma = E_GeV * 1e3 / (physical_constants["electron mass energy equivalent in MeV"][0])    # relativistic Lorentz factor
sigma_x = sigma_y = 15e-6        # fixed beam radius [m]
normalized_emittance = 1e-1        # um-rad (WEAK FOCUSING, if you check the corresponding beta function)
bunch_energy_spread = 1e-4       # initial energy spread
lam_u = 0.04                     # undulator period [m]
K = 2                           # undulator parameter
ku = 2 * math.pi / lam_u
# Scan Ranges
Q_vals = np.logspace(-12, -9, 150)#np.array([50e-12])      # total charge: in Coulombs
Lb_vals = np.logspace(-8.2, -6, 150)#np.array([10e-6])    # bunch length: in meters
###

######################

rho_grid = np.zeros((len(Q_vals), len(Lb_vals)))
P_sat3d_grid = np.zeros_like(rho_grid)
DC_grid = np.zeros_like(rho_grid) # scaled DC detuning (eta_coherent / rho)
Xi_grid = np.zeros_like(rho_grid)

for i, Q in enumerate(Q_vals):
    for j, Lb in enumerate(Lb_vals):
        log.debug("%d, %d", i, j)

        # Particle e.o.m. stats
        transverse_oscillation_amplitude = K / (gamma * ku) * 1e6   # in um.
        log.debug("transverse oscillation amplitude (in um): %.2f", transverse_oscillation_amplitude)

        # JJ factor for planar undulator
        xi = K**2 / (4 + 2 * K**2)
        JJ = float(jv(0, xi) - jv(1, xi))

        # Peak current
        I = Q * c / Lb  # uniform longitudinal profile is assumed
        log.debug("Beam Current (kA) -> %.2f", I / 1e3)

        # Radiation wavelength
        Lambda_r = lam_u / (2 * gamma**2) * (1 + K**2 / 2)   # in meters
        log.debug("Radiation wv. (nm), # of expected microbunches --> %.3f , %d", Lambda_r * 1e9, int(Lb / Lambda_r) )
        log.debug("1st harmonic energy (eV) --> %.2f",   h * c / (Lambda_r * electron_volt))

        # 1D Pierce parameter
        rho_1d_val = rho_1d(I, sigma_x, sigma_y, gamma, lam_u, K, JJ)

        # --- Xie 3D correction (power gain length) ---
        sigma_r = np.sqrt(sigma_x * sigma_y)             # rms size (round-beam proxy)
        epsn = normalized_emittance * 1e-6               # [μm·rad] -> [m·rad]
        sigma_eta = bunch_energy_spread / 1              # keep your slice proxy (1 / 3) or projected bunch spread ( / 1 )
        Lambda_xie, Lg0, Lg3d, rho_eff, rho_perp, xdiag = xie_3d_from_rho(
            lambda_u=lam_u,
            lambda_r=Lambda_r,
            rho_1d=rho_1d_val,
            sigma_r=sigma_r,
            epsn=epsn,
            gamma=gamma,
            sigma_eta=sigma_eta,
            beta_av=None                                # or set a fixed beta here if you prefer
        )
        log.debug("Xie 3D correction --> Lambda, gain lengths (1d - 3d), pierce parameters (1d - 3d):\n %.3f  --  %.3f  --  %.3f  --  %.5f  --  %.5f",
                 Lambda_xie, Lg0, Lg3d, rho_1d_val, rho_eff)
        log.debug("Xie correction parameters --> %.3f  --  %.3f  --  %.3f", xdiag["Xd"], xdiag["Xeps"], xdiag["Xg"])
        log.debug("Matched beta (in meters) --> %.1f", xdiag["beta_av"])

        # Keep full-Xie rho_eff for the existing estimates and LSC metric.
        rho = rho_eff    ### otherwise, 1d FEL value --> rho_1d_val * Correction_Factor
        rho_grid[i, j] = rho

        # Beam power and 3D-corrected saturation power
        P_beam = gamma * m_e * c**2 * (I / e)
        P_sat3d_grid[i, j] = 1.6 * rho_1d_val * P_beam * (Lg0 / Lg3d)**2
        log.debug("Saturation power (GW) --> %.2f", P_sat3d_grid[i, j] / 1e9)
        if rho <= 0:
            Xi_grid[i, j] = np.nan
            continue

        # 3D gain length and interaction length (~saturation length)
        Lg = Lg3d   ### otherwise, 1d approx.  lam_u / (4 * math.pi * math.sqrt(3) * rho)
        log.debug("Gain length (m) --> %.2f", Lg)
        L_int = 18.0 * Lg   # demonstrative saturation distance
        log.debug("Saturation length (m) --> %.2f", L_int)

        #### Longitudinal space-charge (LSC) field from IGF method for a given transverse profile
        reduced_gamma = gamma / np.sqrt(1.0 + K**2 / 2.0)  # averaged longitudinal velocity under planar undulator via ultrarelativistic period average; Eq. (2.15) in FEL book.
        E_0_electrostatic_model = E_z_onaxis(sigma_r, -Q, Lb * reduced_gamma, 1, True, False)
        bunch_duration = len(E_0_electrostatic_model) / 2
        E_0_electrostatic_model_finite_bunch = E_0_electrostatic_model[int(0.5*bunch_duration) : int(1.5*bunch_duration)]
        E_eff = np.mean(np.abs(E_0_electrostatic_model_finite_bunch[:])) # mean longitudinal electric-field
        log.debug("Calculated mean E_z (V/m) --> %.2f", E_eff)

        # Coherent energy detuning over L_int
        eta_coh = e * E_eff * L_int / (gamma * m_e * c**2)
        DC_grid[i, j] = eta_coh / rho

        # Plasma term k_p / Gamma
        n_e = Q / (math.pi * (1.25 * sigma_r)**2 * Lb * e) # uniform-width-equivalance
        omega_p = n_e * e**2 / (epsilon_0 * m_e)
        kp = np.sqrt((1.0 + K**2 / 2.0) * omega_p / (gamma**3 * c**2))
        Gamma = 2.0 * ku * rho
        mu = kp / Gamma if Gamma > 0 else 0.0

        # Cumulative normalized LSC detuning parameter
        Xi = math.sqrt((eta_coh / rho)**2 + mu**2)
        Xi_grid[i, j] = Xi
        log.debug("LSC ingredients (DC, AC, total) --> %.3f , %.3f, %.3f", eta_coh / rho, mu, Xi_grid[i, j])

        # Eigenmodes: transverse-only coupling; spread enters the Gaussian once.
        Gamma_perp = 2.0 * ku * rho_perp
        eta_total = eta_coh  # coherent mean detuning, without an RMS-spread offset
        roots = fel_eigenvalues(eta_total, rho_perp, Gamma_perp, kp=kp,
                               warm_beam=warm_beam, sigma_eta=sigma_eta)
        a_with_LSC = roots[np.argmax(roots.real)] if roots.size else None
        roots = fel_eigenvalues(0.0, rho_perp, Gamma_perp, kp=0.0,
                               warm_beam=warm_beam, sigma_eta=sigma_eta)
        a_wo_LSC = roots[np.argmax(roots.real)] if roots.size else None
        # Zero denotes no resolved positive growth, not an invented eigenvalue.
        g_with_LSC = max(0.0, a_with_LSC.real) if a_with_LSC is not None else 0.0
        g_wo_LSC = max(0.0, a_wo_LSC.real) if a_wo_LSC is not None else 0.0
        log.debug("Eigenvalue problem growing modes --> %.5f, %.5f, %.5f, %.5f, %.5f", rho_perp, sigma_eta, eta_total, g_wo_LSC, g_with_LSC)

        # number of spikes from the cooperation_length / bunch_length
        cooperation_length = Lambda_r / (4*np.pi*rho)
        log.debug("Coop. length (um), bunch length (um) --> %.6f, %.6f", cooperation_length*1e6,  Lb*1e6)
        log.debug("Expected # of SASE spikes at saturation --> %d", max(1, int(np.rint(Lb / (2 * math.pi * cooperation_length)))) )




def get_RaDiO_Spectra(simulation_results_path, post_saturation_end_time, input_kernel_size=15):
    collected_spectra = None
    freqs = None
    collected_power = None
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
        log.info("Simulation problem frequency: {}".format(wr))
        # create diagnostics object
        S = happi.Open(simulation_results_path, reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
        log.info("Happi loads simulation results from the path: " + simulation_results_path)
        ## General info for simulation parameters $ available diagnostics
        dt = S.namelist.Main.timestep
        dx = S.namelist.Main.cell_length[0]
        dy = S.namelist.Main.cell_length[1]
        simulation_geometry = S.namelist.Main.geometry
        simulation_time = S.namelist.Main.simulation_time
        log.info(' Space steps: %f %f'%(dx,dy))
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
                    log.info("Radiation Detector read --> file: {} , rank: {}, datetime: {}".format(os.path.basename(radiation_file), rank, date_time))
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
                    log.info("Collected RaDiO fields --> file: {} , rank: {} , pixel # {}".format(os.path.basename(filename), rank_number, pixel_index))

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
        window_end = int(len(signal_time) * post_saturation_end_time) # len(signal_time)
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
        kernel_size = input_kernel_size
        kernel = np.ones(kernel_size) / kernel_size
        power_spectral_density[0:HF_limit] = np.convolve(power_spectral_density[0:HF_limit], kernel, mode='same')
        np.set_printoptions(threshold=np.inf)
        print(simulation_results_path)
        collected_spectrum = power_spectral_density[0:HF_limit]
        freqs = frequencies[0:HF_limit] * relativistic_doppler_correction_factor / scipy.constants.physical_constants["electron volt-hertz relationship"][0]
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
        log.info("Parseval (DFT) check --> parseval_t: {} , parseval_w: {} , fraction: {}".format(parseval_t, parseval_w, parseval_t/parseval_w))

        collected_power = temporal_intensity
        return freqs, collected_spectrum, collected_power
    else:
        return None



###
def get_bunching_factors(simulation_folders):
    def phases_from_z(z, k):
        """Return phases θ = (k z) mod 2π for an array of positions z."""
        return np.mod(k * z, 2*np.pi)

    def bunching_factor(phases, n=1):
        """Compute |(1/N) sum exp(i n theta)| for given phases theta in radians."""
        return np.abs(np.mean(np.exp(1j * n * phases)))


    bunching_factor_statistics_dict = {}
    for folder_id, simulation_folder in enumerate(simulation_folders):
        simulation_path = "/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/" + simulation_folder + "/simulation_results/"
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            # read log-file
            simulation_parameters = {} # to keep important simulation parameters to be used in diagnostics, such as wr for proper unit conversion
            with open(simulation_path + "/simulation_parameters.txt") as f:
                for l in f:
                    line_data = l.strip().replace(" ", "").split('=')
                    simulation_parameters[line_data[0]] = float(line_data[1])
            print(simulation_parameters)
            wr = None # can be manually set here too
            if "wr" in simulation_parameters:
                wr = simulation_parameters["wr"]
            logging.info("Simulation problem frequency: {}".format(wr))
            # create diagnostics object
            S = happi.Open(simulation_path, reference_angular_frequency_SI=wr, show=True, verbose=True, pint=True, scan=False)
            logging.info("Happi loads simulation results from the path: " + simulation_path)
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
            um  = 1.e-6 * wr / c            # 1 micrometer in normalized units
            radiation_wavelength_in_labframe = simulation_parameters["FEL_radiation_wl"] / um
            radiation_wavelength_in_bunchrestframe = 2 * reference_frame_gamma * radiation_wavelength_in_labframe  # \lambda_r'prime
            microbunching_interval_in_bunchrestframe = reference_frame_gamma * radiation_wavelength_in_labframe # \lambda_b'prime
            print("Radiation wl in lab frame (um) --> ", radiation_wavelength_in_labframe)
            print("Radiation wl in bunch rest frame (um) & frame gamma --> ", radiation_wavelength_in_bunchrestframe, reference_frame_gamma)
            k_r = 2 * np.pi / radiation_wavelength_in_bunchrestframe  # forward radiation wavenumber in the simulation frame
            k_b = 2 * np.pi / microbunching_interval_in_bunchrestframe  # microbunching wavenumber in the simulation frame

            timesteps = S.TrackParticles(species="electron", axes=["x"], sort=False, units=unit_dictionary["units"]).getTimesteps().astype(int)
            start_time_index=25
            end_time_index=-12
            print("Timesteps for the bunch profile data --> ", len(timesteps), timesteps)
            colors = plt.colormaps["coolwarm"](np.linspace(0, 1, len(timesteps[start_time_index:end_time_index])))
            harmonic_id = int(1) # 1, 3, 5, .. --> odd harmonics
            bn_mean = []
            bn_std = []
            for timestep_id, timestep in enumerate(timesteps[start_time_index:end_time_index]):
                bunching_factors = []
                microbunch_numbers = 20   # number of microbunches [SHOULD BE BALANCED]
                slice_length = microbunch_numbers * microbunching_interval_in_bunchrestframe / harmonic_id  # slice length in bunch rest-frame (in um.)
                top = 370
                bottom = 350
                diagnostics_axes = ["moving_x", "x", "y", "z", "px", "py", "pz", "w"]
                ParticleTrackDiag = S.TrackParticles(species="electron", timesteps=timestep, axes=diagnostics_axes, sort=False, units=unit_dictionary["units"]) # for all enabled species within this tracking diagnostics
                track_data = ParticleTrackDiag.getData()[timestep] # axis-based entries within a dictionary returned for a selected timestep (oth index for the specific timestep diagnostics, not cumulative)
                weights = track_data["w"]
                x = track_data["x"]
                y = track_data["y"]
                z = track_data["z"]
                print(timestep, "Diagnostics is read with size: ", len(x))
                indexes = np.arange(0, len(weights), dtype=int)  # default indexes
                transformed_axis = x - np.min(x)  # start from 0
                normalized_transverse_axis_y = (y - bottom) / (top - bottom)
                normalized_transverse_axis_z = (z - bottom) / (top - bottom)
                slice_sections = np.arange(0, 0.96*(np.max(x) - np.min(x)), slice_length)
                for x_start in slice_sections:
                    conditions = [transformed_axis > x_start,
                                  transformed_axis < (x_start + slice_length),
                                  normalized_transverse_axis_y <= 1,
                                  normalized_transverse_axis_y >= 0,
                                  normalized_transverse_axis_z <= 1,
                                  normalized_transverse_axis_z >= 0
                                  ]
                    selected_indexes = np.logical_and.reduce(conditions)
                    z_position_holder = transformed_axis[selected_indexes]
                    z_position_holder = z_position_holder.flatten()
                    #print(len(z_position_holder))
                    ##  bunching factor calculation
                    b_n = bunching_factor(phases_from_z(z_position_holder, harmonic_id * k_b))
                    #print("bunching factor -> ", b_n)
                    bunching_factors.append(b_n)
                bn_mean.append(np.mean(bunching_factors)) # profile-based, so for each timestep
                bn_std.append(np.std(bunching_factors)) # profile-based, so for each timestep
                ax.plot(bunching_factors, color=colors[timestep_id], linewidth=2)
            print(bn_mean, bn_std) # for statistical evolution of bunching factor profile
            bunching_factor_statistics_dict[simulation_folder + "_mean"] = bn_mean
            bunching_factor_statistics_dict[simulation_folder + "_std"] = bn_std

            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            from matplotlib.ticker import FixedLocator, NullFormatter
            #ax.set_ylim(0, 0.35)
            ax.tick_params(axis="both", which="major", labelsize=25, length=8, width=1.5, direction="out")
            fig.tight_layout()
            fig.savefig(simulation_folder + "_bunchingfactor_profiles.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        except Exception as error:
            print("Problem in the microbunch profile extraction: ", error)


    #### plot evolution of bunching profile statistics for both simulations
    from matplotlib.ticker import MaxNLocator
    mean1 = np.array(bunching_factor_statistics_dict[simulation_folders[0]+"_mean"][:])
    std1 = np.array(bunching_factor_statistics_dict[simulation_folders[0]+"_std"][:])
    mean2 =  np.array(bunching_factor_statistics_dict[simulation_folders[1]+"_mean"][:])
    std2 =  np.array(bunching_factor_statistics_dict[simulation_folders[1]+"_std"][:])
    step=np.arange(1,len(mean1)+1)
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(3,1,figsize=(10.5,11.5),sharex=True,gridspec_kw={'height_ratios':[2.2,1,1.1]}, layout='constrained')
    ax=axes[0]
    for avg,sd,label,color in [(mean1,std1,'Profile 1','tab:blue'),(mean2,std2,'Profile 2','tab:orange')]:
        ax.fill_between(step,avg-sd,avg+sd,color=color,alpha=.16,linewidth=0)
        ax.plot(step,avg,color=color,lw=2.3,label=label)
    ax.set_ylabel('Profile value')
    ax.set_title('Mean evolution with ±1 standard deviation',weight='bold')
    ax.legend(frameon=False,ncol=2)
    ax.grid(alpha=.2)
    ax.text(.01,.03,'Shaded envelopes: mean ± SD (not confidence intervals)',transform=ax.transAxes,fontsize=9)
    ax=axes[1]
    ax.plot(step,std1,color='tab:blue',lw=2,label='Profile 1')
    ax.plot(step,std2,color='tab:orange',lw=2,label='Profile 2')
    ax.set_ylabel('Within-step SD')
    ax.set_title('Evolution of variability',loc='left')
    ax.grid(alpha=.2)
    ax=axes[2]
    ax.plot(step,mean2-mean1,color='tab:purple',lw=2.3)
    ax.axhline(0,color='0.3',lw=1,ls='--')
    ax.fill_between(step,0,mean2-mean1,color='tab:purple',alpha=.12)
    ax.set_ylabel('Mean₂ − Mean₁')
    ax.set_xlabel('Evolution step (index)')
    ax.set_title('Difference between profile means',loc='left')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=10))
    ax.grid(alpha=.2)
    fig.savefig(simulation_folder + "bunching_factor_profile_evolution.png",dpi=220,bbox_inches='tight')
    plt.close(fig)
    # Distribution-style comparison: mean±sd at selected steps (normal approximations, not measured histograms).
    from scipy.stats import norm
    select=[0,9,19,29,38]
    fig,ax=plt.subplots(figsize=(10.5,5.6),layout='constrained')
    xx=np.linspace(-.08,.65,850)
    from matplotlib import colormaps
    cmap=colormaps['viridis']
    for j,i in enumerate(select):
        color=cmap(j/(len(select)-1))
        ax.plot(xx,norm.pdf(xx,mean1[i],std1[i]),ls='-',color=color,lw=2,label=f'Step {i+1}')
        ax.plot(xx,norm.pdf(xx,mean2[i],std2[i]),ls='--',color=color,lw=2)
    ax.set_xlabel('Profile value');ax.set_ylabel('Probability density (illustrative)')
    ax.set_title('Gaussian approximations from mean ± SD at selected steps',weight='bold')
    from matplotlib.lines import Line2D
    l1=ax.legend(title='Evolution step',frameon=False,loc='upper right')
    ax.add_artist(l1)
    ax.legend(handles=[Line2D([0],[0],color='black',lw=2,label='Profile 1 — solid'),Line2D([0],[0],color='black',lw=2,ls='--',label='Profile 2 — dashed')],frameon=False,loc='upper left')
    ax.grid(alpha=.15)
    fig.savefig(simulation_folder + "bunching_factor_profile_distribution.png", dpi=220, bbox_inches='tight')
    plt.close(fig)
    plt.rcdefaults()


#########################################################################################################################################################


# -----------------------------------------------------------------------------
# 2x2 hardXray figure: subfigures give each panel its own layout and colorbar.
# -----------------------------------------------------------------------------


def style_axis(ax, labelsize=30, ticksize=30):
    ax.tick_params(axis='both', which='major', labelsize=ticksize,
                   length=8, width=1.5, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.grid(False)


def plot_heatmap(subfig, ax):
    """Heatmap for hardXray setups besides the LSC metric curves."""
    x, y = np.meshgrid(Lb_vals * 1e6, Q_vals * 1e12)
    hm = ax.pcolormesh(x, y, P_sat3d_grid / 1e9,
                       cmap='plasma', norm=LogNorm(vmin=1e2, vmax=2e6),
                       shading='auto')
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$L_{\mathrm{spike}}\,(\mu\mathrm{m})$', fontsize=30)
    ax.set_ylabel(r'$Q_{\mathrm{spike}}\,(\mathrm{pC})$', fontsize=30)
    style_axis(ax)
    ax.minorticks_off()

    # Create colorbar immediately beside the heatmap
    cax = ax.inset_axes( [1.01, 0.0, 0.045, 1.0], transform=ax.transAxes )
    cb = subfig.colorbar(
        hm,
        cax=cax,
        orientation="vertical"
    )
    cb.set_label( r"$\mathrm{P}_{\mathrm{sat}}\ (\mathrm{GW})$", fontsize=27, labelpad=1 )
    cb.set_ticks([1e3, 1e4, 1e5, 1e6])
    cb.set_ticklabels([rf"$10^{{{n}}}$" for n in range(3, 7)])
    cb.ax.tick_params(length=5, width=1.1, labelsize=24, pad=1)

    cs = ax.contour(x, y, Xi_grid, levels=[0.45, 1.0, 2.0],
                    linewidths=4,
                    colors=['lightskyblue', 'springgreen', 'crimson'],
                    linestyles='--')
    handles, _ = cs.legend_elements()
    labels = [rf'$\Lambda_{{LSC}}\!=\!{level:g}$' for level in [0.4, 1, 2]]


    ax.text(0.645, 0.63, r'$\mathrm{E_{b}\!\approx\!17\,GeV}$',
            fontsize=21, color="beige", transform=ax.transAxes, zorder=50)
    ax.text(0.645, 0.56, r'$\mathrm{\sigma_{\!\perp}\!\approx\!15\, \mu m}$',
            fontsize=21, color="beige", transform=ax.transAxes, zorder=50)
    ax.text(0.645, 0.49, r'$\mathrm{K\!=\!2}$',
            fontsize=21, color="beige", transform=ax.transAxes, zorder=50)

    star_red = ax.scatter([0.01], [34], marker='*', s=500,
               color='crimson', edgecolor='black', linewidth=0.8, zorder=10000)
    star_blue = ax.scatter([0.06], [4], marker='*', s=500,
               color='deepskyblue', edgecolor='black', linewidth=0.8, zorder=10000)
    handles.extend([star_red, star_blue])
    labels.extend([
        r'$\mathrm{1\, MA}$',
        r'$\mathrm{20\, kA}$',
    ])

    ax.legend(handles, labels, loc=(0.58, 0.01), fontsize=23, handletextpad=0.2, handlelength=0.75, labelspacing=0.3,
              framealpha=0.2, borderpad=0.10, labelcolor="beige")

    ax.text(0.025, 0.975, '(a)', transform=ax.transAxes,
            fontsize=27, fontweight='bold', va='top')
    return hm


def plot_20kA_curve(subfig, ax):
    axins = ax.inset_axes([0.53, 0.58, 0.46, 0.41])
    """Power spectrum and growth for the blue star"""
    simulation_folders = ["draft_hardXray_60nm_ep_saturation", "draft_hardXray_60nm_e_only_saturation"]
    post_saturation_end_times = [0.99, 0.99]
    labels = [r"$e^- \, / \,\, e^+$", r"$e^-$"]
    colors = ["#00CFFF", "navy"]
    linewidths = [5.0, 3.5]
    linestyles = ["-", "--"]
    alpha_values= [1.0, 0.85]
    # get the data
    main_freq = None
    for folder_id, folder_name in enumerate(simulation_folders):
        simulation_path = "/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/" + folder_name + "/simulation_results/"
        post_saturation_end_time = post_saturation_end_times[folder_id]
        if pathlib.Path(simulation_path + "RaDiO_data.npz").is_file():
            with np.load(simulation_path + "RaDiO_data.npz") as data:
                freqs = data["freqs"]
                collected_spectrum = data["collected_spectrum"]
                collected_power = data["collected_power"]
                print("RaDiO data is read directly from saved npz file for the simulation: " + folder_name)
        else:
            freqs, collected_spectrum, collected_power = get_RaDiO_Spectra(simulation_path, post_saturation_end_time)
            np.savez_compressed(simulation_path + "RaDiO_data.npz", freqs=freqs, collected_spectrum=collected_spectrum, collected_power=collected_power)
            print("RaDiO data is created online and written to the npz file for the simulation: " + folder_name)
        # set the 1st harmonic frequency
        if main_freq is None:
            main_freq = freqs[np.argmax(np.array(collected_spectrum))] / 1000 # in keV

        # plot the odd harmonics
        ax.plot(freqs/1000, collected_spectrum, color=colors[folder_id],
                linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], label=labels[folder_id], alpha=alpha_values[folder_id], zorder=3)

        # plot the smoothed growth curve
        power = np.asarray(collected_power)
        block = 800
        n = len(power) // block
        power_mean = power[:n * block].reshape(n, block).mean(axis=1) # Average every 1000 samples
        power_smooth = gaussian_filter1d( power_mean, sigma=20, mode="nearest" ) # Gaussian smoothing
        z_smooth = np.linspace(0, 1, n) # Corresponding normalized longitudinal coordinate
        axins.plot(z_smooth, power_smooth, color=colors[folder_id], linewidth=3, linestyle=linestyles[folder_id], alpha=alpha_values[folder_id] )
        #axins.plot(collected_power, color=colors[folder_id], linewidth=0.5, linestyle=linestyles[folder_id] )

    # axis esthetics
    print("w_0 (in keV) --> ", main_freq)
    style_axis(ax)
    ax.minorticks_off()
    ax.set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=30)
    ax.set_xlabel(r"keV", fontsize=30)
    ax.set_xlim(0.7 * main_freq, 5.5*main_freq)
    ax.set_ylim(1e20, 2e24)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.legend(fontsize=24, loc=(0.13, 0.78), frameon=True, markerscale=180, handletextpad=0.2, handlelength=1.0, labelspacing=0.3, borderpad=0.10)
    ax.set_yticks([1e21, 1e22, 1e23, 1e24])
    ax.set_yticklabels([r"$10^{21}$", r"$10^{22}$", r"$10^{23}$", r"$10^{24}$"])
    ax.set_xticks([20, 50, 80, 110])
    ax.set_xticklabels(["20", "50", "80", "110"])
    ax.text(0.01, 0.985, '(b)', transform=ax.transAxes,
            fontsize=27, fontweight='bold', va='top')
    ax.text(0.15, 0.74, '20 kA', transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', color="dodgerblue")

    # sub-panel for the power growth
    axins.set_facecolor("#F2F2F2")
    axins.set_xscale("linear")
    axins.set_yscale("log")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlabel(r"z [a.u.]", fontsize=20)
    axins.set_ylabel(r"P [a.u.]", fontsize=20, labelpad=-10)
    for spine in axins.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)
    axins.grid(False)


    ### get bunching factor for high-harmonics
    #get_bunching_factors(simulation_folders)



def plot_1MA_curve(subfig, ax):
    """Power spectrum and growth for the red star"""
    axins = ax.inset_axes([0.53, 0.63, 0.46, 0.36])
    """Power spectrum and growth for the blue star"""
    simulation_folders = ["draft_hardXray_10nm_ep_2", "draft_hardXray_10nm_e_only_2"]
    post_saturation_end_times = [0.99, 0.99]
    labels = [r"$e^- \, / \,\, e^+$", r"$e^-$"]
    colors = ["coral", "maroon",]
    #colors = ["#CC0033", "#FF9671"]
    linewidths = [5.0, 3.5]
    linestyles = ["-", "--"]
    alpha_values= [1.0, 0.85]
    # get the data
    main_freq = None
    for folder_id, folder_name in enumerate(simulation_folders):
        simulation_path = "/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/" + folder_name + "/simulation_results/"
        post_saturation_end_time = post_saturation_end_times[folder_id]
        if pathlib.Path(simulation_path + "RaDiO_data.npz").is_file():
            with np.load(simulation_path + "RaDiO_data.npz") as data:
                freqs = data["freqs"]
                collected_spectrum = data["collected_spectrum"]
                collected_power = data["collected_power"]
                print("RaDiO data is read directly from saved npz file for the simulation: " + folder_name)
        else:
            freqs, collected_spectrum, collected_power = get_RaDiO_Spectra(simulation_path, post_saturation_end_time)
            np.savez_compressed(simulation_path + "RaDiO_data.npz", freqs=freqs, collected_spectrum=collected_spectrum, collected_power=collected_power)
            print("RaDiO data is created online and written to the npz file for the simulation: " + folder_name)
        # set the 1st harmonic frequency
        if main_freq is None:
            main_freq = freqs[np.argmax(np.array(collected_spectrum))] / 1000 # in keV

        # plot the odd harmonics
        ax.plot(freqs/1000, collected_spectrum, color=colors[folder_id],
                linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], label=labels[folder_id], alpha=alpha_values[folder_id], zorder=3)

        # plot the smoothed growth curve
        power = np.asarray(collected_power)
        block = 800
        n = len(power) // block
        power_mean = power[:n * block].reshape(n, block).mean(axis=1) # Average every 1000 samples
        power_smooth = gaussian_filter1d( power_mean, sigma=20, mode="nearest" ) # Gaussian smoothing
        z_smooth = np.linspace(0, 1, n) # Corresponding normalized longitudinal coordinate
        axins.plot(z_smooth, power_smooth, color=colors[folder_id], linewidth=3, linestyle=linestyles[folder_id], alpha=alpha_values[folder_id] )
        #axins.plot(collected_power, color=colors[folder_id], linewidth=0.5, linestyle=linestyles[folder_id] )

    # axis esthetics
    print("w_0 (in keV) --> ", main_freq)
    style_axis(ax)
    ax.minorticks_off()
    ax.set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=30)
    ax.set_xlabel(r"keV", fontsize=30)
    ax.set_xlim(0.7 * main_freq, 5.5*main_freq)
    ax.set_ylim(1e22, 4e28)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.legend(fontsize=24, loc=(0.13, 0.78), frameon=True, markerscale=180, handletextpad=0.2, handlelength=1.0, labelspacing=0.3, borderpad=0.10)
    ax.set_yticks([1e23, 1e25, 1e27])
    ax.set_yticklabels([r"$10^{23}$", r"$10^{25}$", r"$10^{27}$"])
    ax.set_xticks([20, 50, 80, 110])
    ax.set_xticklabels(["20", "50", "80", "110"])
    ax.text(0.01, 0.985, '(c)', transform=ax.transAxes,
            fontsize=27, fontweight='bold', va='top')
    ax.text(0.15, 0.74, '1 MA', transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', color="crimson")

    # sub-panel for the power growth
    axins.set_facecolor("#F2F2F2")
    axins.set_xscale("linear")
    axins.set_yscale("log")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlabel(r"z [a.u.]", fontsize=20)
    axins.set_ylabel(r"P [a.u.]", fontsize=20, labelpad=-10)
    for spine in axins.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.8)
    axins.grid(False)



def plot_robustbess_1stHarmonic(subfig, ax):
    """Robustness tests of the red star via overlapping and total charge mismatch"""
    ax.axis("off") # Hide the original parent axes
    style_axis(ax)
    ax_left = ax.inset_axes([0.00, 0.00, 0.50, 1.00])
    ax_right = ax.inset_axes([0.50, 0.00, 0.50, 1.00], sharey=ax_left) # Create the left and right axes
    style_axis(ax_left)
    style_axis(ax_right)
    ax_right.tick_params(axis="y", which="both",
                         left=False,
                         labelleft=False, ) # Hide y-axis ticks and labels on right plot
    ax_left.spines["right"].set_visible(True) # Only one vertical line between the plots
    ax_right.spines["left"].set_visible(False)
    ax_left.spines["right"].set_color("#888888")
    ax_left.spines["right"].set_linewidth(1.8)
    ax_right.spines["left"].set_visible(False)
    

    ############ Charge imbalance
    simulation_folders = ["draft_hardXray_10nm_ep_2", "draft_hardXray_10nm_ep_Q09", "draft_hardXray_10nm_ep_Q08", "draft_hardXray_10nm_ep_Q07", "draft_hardXray_10nm_ep_Q065", ]
    post_saturation_end_times = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    colors = [
    "blanchedalmond",  # 0%
    "#FFD68A",  # 10%
    "#FFBB55",  # 20%
    "#F47700",  # 25%
    "#B83B00", # 30%
    ]
    labels = ["ideal", "10%", "20%", "30%", "35%"]
    linewidths = [2.8, 2.5, 2.1, 2.0, 1.8,]
    linestyles = ["-", "-", "-", "-", "-",]
    # get the data
    collected_spectra = []
    collected_freqs = []
    main_freq = None
    spectrum_kernel_size = int(4)
    for folder_id, folder_name in enumerate(simulation_folders):
        simulation_path = "/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/" + folder_name + "/simulation_results/"
        post_saturation_end_time = post_saturation_end_times[folder_id]
        if pathlib.Path(simulation_path + "RaDiO_data.npz").is_file():
            with np.load(simulation_path + "RaDiO_data.npz") as data:
                freqs = data["freqs"]
                collected_spectrum = data["collected_spectrum"]
                collected_power = data["collected_power"]
                print("RaDiO data is read directly from saved npz file for the simulation: " + folder_name)
        else:
            freqs, collected_spectrum, collected_power = get_RaDiO_Spectra(simulation_path, post_saturation_end_time, spectrum_kernel_size)
            np.savez_compressed(simulation_path + "RaDiO_data.npz", freqs=freqs, collected_spectrum=collected_spectrum, collected_power=collected_power)
            print("RaDiO data is created online and written to the npz file for the simulation: " + folder_name)
        # set the 1st harmonic frequency
        if main_freq is None:
            main_freq = freqs[np.argmax(np.array(collected_spectrum))] / 1000 # in keV

        # plot the odd harmonics
        ax_left.fill_between(freqs/1000, collected_spectrum, y2=1e25, color=colors[folder_id], alpha=1.0, zorder=1+folder_id)
        ax_left.plot(freqs/1000, collected_spectrum, color=colors[folder_id], linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], alpha=0.06, label=labels[folder_id], zorder=10 + folder_id)

    # axis esthetics
    print("w_0 (in keV) --> ", main_freq)

    ax_left.set_xscale("linear")
    ax_left.set_yscale("log")
    ax_left.set_xlabel(r"keV", fontsize=30)
    ax_left.set_ylabel(r"$d\varepsilon \, / \, d\omega d\Omega$ [a.u.]", fontsize=30)
    ax_left.set_xlim(23.500, 24.140)
    ax_left.set_ylim(1e25, 1e28)
    ax_left.set_yticks([1e25, 1e26, 1e27, 1e28])
    ax_left.set_yticklabels([r"$10^{25}$", r"$10^{26}$", r"$10^{27}$", r"$10^{28}$"])
    freqs_array = [23.7, 24.0]
    freqs_labels = [str(x-1.) for x in freqs_array]
    ax_left.set_xticks(freqs_array)
    ax_left.set_xticklabels(freqs_labels)

    ax_left.text( 0.02, 0.90, r"$\boldsymbol{\Delta Q\!=\!0\%}$", fontsize=20, color="slategray", transform=ax_left.transAxes, zorder=100, )
    ax_left.text( 0.02, 0.75, r"$\boldsymbol{\Delta Q\!=\!10\%}$", fontsize=20, color="slategray", fontweight="bold", transform=ax_left.transAxes, zorder=100, )
    ax_left.text( 0.02, 0.59, r"$\boldsymbol{\Delta Q\!=\!20\%}$", fontsize=20, color="slategray", transform=ax_left.transAxes, zorder=100, )
    ax_left.text( 0.02, 0.42, r"$\boldsymbol{\Delta Q\!=\!30\%}$", fontsize=20, color="slategray", transform=ax_left.transAxes, zorder=100, )
    ax_left.text( 0.02, 0.19, r"$\boldsymbol{\Delta Q\!=\!35\%}$", fontsize=20, color="slategray", transform=ax_left.transAxes, zorder=100, )

    ax_left.text(0.36, 0.985, '(d)', transform=ax.transAxes,
            fontsize=27, fontweight='bold', va='top')



    ############ Charge imbalance
    simulation_folders = ["draft_hardXray_10nm_ep_2", "draft_hardXray_10nm_ep_OverlapShift005", "draft_hardXray_10nm_ep_OverlapShift01", "draft_hardXray_10nm_ep_OverlapShift015", "draft_hardXray_10nm_ep_OverlapShift02", ]
    post_saturation_end_times = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    colors = [
    "#DADAEB",  # 0%
    "#BCBDDC",  # 5%
    "#9E9AC8",  # 10%
    "#807DBA",  # 15%
    "#54278F", # 20%
    ]
    labels = ["ideal", "5%", "10%", "15%", "20%"]
    linewidths = [2.8, 2.5, 2.1, 2.0, 1.8,]
    linestyles = ["-", "-", "-", "-", "-",]
    # get the data
    collected_spectra = []
    collected_freqs = []
    main_freq = None
    spectrum_kernel_size = int(4)
    for folder_id, folder_name in enumerate(simulation_folders):
        simulation_path = "/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/" + folder_name + "/simulation_results/"
        post_saturation_end_time = post_saturation_end_times[folder_id]
        if pathlib.Path(simulation_path + "RaDiO_data.npz").is_file():
            with np.load(simulation_path + "RaDiO_data.npz") as data:
                freqs = data["freqs"]
                collected_spectrum = data["collected_spectrum"]
                collected_power = data["collected_power"]
                print("RaDiO data is read directly from saved npz file for the simulation: " + folder_name)
        else:
            freqs, collected_spectrum, collected_power = get_RaDiO_Spectra(simulation_path, post_saturation_end_time, spectrum_kernel_size)
            np.savez_compressed(simulation_path + "RaDiO_data.npz", freqs=freqs, collected_spectrum=collected_spectrum, collected_power=collected_power)
            print("RaDiO data is created online and written to the npz file for the simulation: " + folder_name)
        # set the 1st harmonic frequency
        if main_freq is None:
            main_freq = freqs[np.argmax(np.array(collected_spectrum))] / 1000 # in keV

        # plot the odd harmonics
        ax_right.fill_between(freqs/1000, collected_spectrum, y2=1e25, color=colors[folder_id], alpha=1.0, zorder=1+folder_id)
        ax_right.plot(freqs/1000, collected_spectrum, color=colors[folder_id], linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], alpha=0.06, label=labels[folder_id], zorder=10 + folder_id)

    # axis esthetics
    print("w_0 (in keV) --> ", main_freq)
    ax_right.set_xscale("linear")
    ax_right.set_yscale("log")
    ax_right.set_xlabel(r"keV", fontsize=30)
    ax_right.set_xlim(23.550, 24.100)
    ax_right.set_ylim(1e25, 1e28)
    ax_right.set_yticks([1e25, 1e26, 1e27, 1e28])
    ax_right.set_xticks(freqs_array)
    ax_right.set_xticklabels(freqs_labels)

    ax_right.text( 0.6, 0.90, r"$\boldsymbol{\delta\!=\!0\%}$", fontsize=20, color="darkolivegreen", transform=ax_right.transAxes, zorder=100, )
    ax_right.text( 0.6, 0.67, r"$\boldsymbol{\delta\!=\!5\%}$", fontsize=20, color="darkolivegreen", transform=ax_right.transAxes, zorder=1000, )
    ax_right.text( 0.6, 0.56, r"$\boldsymbol{\delta\!=\!10\%}$", fontsize=20, color="darkolivegreen", transform=ax_right.transAxes, zorder=1000, )
    ax_right.text( 0.6, 0.40, r"$\boldsymbol{\delta\!=\!15\%}$", fontsize=20, color="darkolivegreen", transform=ax_right.transAxes, zorder=1000, )
    ax_right.text( 0.6, 0.25, r"$\boldsymbol{\delta\!=\!20\%}$", fontsize=20, color="darkolivegreen", transform=ax_right.transAxes, zorder=1000000, )

    ax_right.text(0.530, 0.985, '(e)', transform=ax.transAxes,
            fontsize=27, fontweight='bold', va='top')




    ###### add positron quality-degraded curves
    simulation_folders = ["draft_hardXray_10nm_ep_2_quality_degraded", ]
    post_saturation_end_times = [0.99]
    colors = [
    "black",
    ]
    labels = []
    linewidths = [6.0]
    linestyles = ["--"]
    # get the data
    collected_spectra = []
    collected_freqs = []
    main_freq = None
    spectrum_kernel_size = int(2)
    for folder_id, folder_name in enumerate(simulation_folders):
        simulation_path = "/lfs/l8/theo/quantumplasma/simulations/Cagri/Smilei/undulator/POC_LorentzBoosted/" + folder_name + "/simulation_results/"
        post_saturation_end_time = post_saturation_end_times[folder_id]
        if pathlib.Path(simulation_path + "RaDiO_data.npz").is_file():
            with np.load(simulation_path + "RaDiO_data.npz") as data:
                freqs = data["freqs"]
                collected_spectrum = data["collected_spectrum"]
                collected_power = data["collected_power"]
                print("RaDiO data is read directly from saved npz file for the simulation: " + folder_name)
        else:
            freqs, collected_spectrum, collected_power = get_RaDiO_Spectra(simulation_path, post_saturation_end_time, spectrum_kernel_size)
            np.savez_compressed(simulation_path + "RaDiO_data.npz", freqs=freqs, collected_spectrum=collected_spectrum, collected_power=collected_power)
            print("RaDiO data is created online and written to the npz file for the simulation: " + folder_name)
        # set the 1st harmonic frequency
        if main_freq is None:
            main_freq = freqs[np.argmax(np.array(collected_spectrum))] / 1000 # in keV

        # plot the odd harmonics
        #ax_left.fill_between(freqs/1000, collected_spectrum, y2=1e25, color=colors[folder_id], alpha=1.0, zorder=3)
        #ax_left.plot(freqs/1000, np.full(len(freqs), np.max(collected_spectrum)), color=colors[folder_id], linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], zorder=1000)
        ax_left.plot(freqs/1000, collected_spectrum, color=colors[folder_id], linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], zorder=50)

        #ax_right.fill_between(freqs/1000, collected_spectrum, y2=1e25, color=colors[folder_id], alpha=1.0, zorder=3)
        #ax_right.plot(freqs/1000, np.full(len(freqs), np.max(collected_spectrum)), color=colors[folder_id], linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], zorder=1000)
        ax_right.plot(freqs/1000, collected_spectrum, color=colors[folder_id], linewidth=linewidths[folder_id], linestyle=linestyles[folder_id], zorder=50)

    # axis esthetics
    print("w_0 (in keV) --> ", main_freq)






def make_2x2_figure():
    # A large physical canvas preserves the original 17-point labels.
    fig = plt.figure(figsize=(14.5, 11.5), layout='constrained')

    # Independent nested layouts: each cell has its own figure-like area,
    # meaning each axes/cbar can be managed locally.
    subfig = fig.subfigures(nrows=2, ncols=2,
                            wspace=0.025, hspace=0.015)
    axs = np.empty((2, 2), dtype=object)
    for ij, sf in np.ndenumerate(subfig):
        axs[ij] = sf.subplots()

    plot_heatmap(subfig[0, 0], axs[0, 0])
    plot_20kA_curve(subfig[0, 1], axs[0, 1])
    plot_1MA_curve(subfig[1, 0], axs[1, 0])
    plot_robustbess_1stHarmonic(subfig[1, 1], axs[1, 1])

    # DO NOT call plt.tight_layout(): it competes with constrained layout.
    return fig, axs, subfig



#########################################################################################################################################################

fig, axs, subfig = make_2x2_figure()
outdir = pathlib.Path(__file__).resolve().parent
pathlib.Path(figname + ".png").unlink(missing_ok=True)
fig.savefig(outdir / (figname + ".png"), format="png", dpi=300, facecolor='white')
plt.close(fig)
