import numpy as np
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import erf
from scipy.interpolate import RectBivariateSpline
from numba import jit

warnings.simplefilter("error", OptimizeWarning)

LYA_WAVELENGTH = 1215.67
LYA_FIRST_WVL = 1050.
LYA_LAST_WVL = 1180.
LYA_CENTER_WVL = (LYA_LAST_WVL + LYA_LAST_WVL) / 2

Si4_FIRST_WVL = 1268.
Si4_LAST_WVL = 1380.

C4_FIRST_WVL = 1409.
C4_LAST_WVL = 1523.

LIGHT_SPEED = 299792.458
ONE_SIGMA_2_FWHM = 2.35482004503


def formBins(nblin, nblog, dklin, dklog, k0, klast=-1):
    lin_bin_edges = np.arange(nblin + 1) * dklin + k0
    log_bin_edges = np.power(
        10., np.arange(1, nblog + 1) * dklog) * lin_bin_edges[-1]

    # assert log_bin_edges[-1] < k_values[-1]

    bin_edges = np.concatenate((lin_bin_edges, log_bin_edges))
    if klast > bin_edges[-1]:
        bin_edges.append(klast)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    return bin_edges, bin_centers


# -----------------------------------------------------
# Power spectrum begins
# -----------------------------------------------------
PDW_FIT_AMP = 6.62141965e-02
PDW_FIT_N = -2.68534876e+00
PDW_FIT_APH = -2.23276251e-01
PDW_FIT_B = 3.59124427e+00
PDW_FIT_BETA = -1.76804541e-01
PDW_FIT_LMD = 3.59826056e+02

PDW_FIT_PARAMETERS = (
    PDW_FIT_AMP, PDW_FIT_N, PDW_FIT_APH, PDW_FIT_B, PDW_FIT_BETA, PDW_FIT_LMD)
PDW_FIT_PARAMETERS_0BETA = (
    PDW_FIT_AMP, PDW_FIT_N, PDW_FIT_APH, PDW_FIT_B, 0, PDW_FIT_LMD)
PD13_PIVOT_K = 0.009
PD13_PIVOT_Z = 3.0


def evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd):
    k, z = X
    q0 = k / PD13_PIVOT_K + 1e-10

    result = (A * np.pi / PD13_PIVOT_K) * np.power(
        q0, 2. + n + alpha * np.log(q0)) / (1. + lmd * k**2)

    if z is not None:
        x0 = (1. + z) / (1. + PD13_PIVOT_Z)
        result *= np.power(q0, beta * np.log(x0)) * np.power(x0, B)

    return result


def evaluatePD13W17Fit(k, z=None):
    if z is None:
        p_noredshif = list(PDW_FIT_PARAMETERS)
        p_noredshif[3] = 0
        p_noredshif[4] = 0

        pfid = p_noredshif
        z = 3.0
    else:
        pfid = list(PDW_FIT_PARAMETERS_0BETA)

    return evaluatePD13Lorentz((k, z), *pfid)


def jacobianPD13Lorentz(X, A, n, alpha, B, beta, lmd):
    k, z = X
    pkz = evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd)
    lnk = np.log(k / PD13_PIVOT_K + 1e-10)

    col_A = pkz / A
    col_n = pkz * lnk
    col_alpha = pkz * lnk * lnk
    col_lmd = -pkz * k**2 / (1. + lmd * k**2)

    result = np.column_stack((col_A, col_n, col_alpha))

    if z is not None:
        lnz = np.log((1. + z) / (1. + PD13_PIVOT_Z))
        col_B = pkz * lnz
        col_beta = pkz * lnk * lnz
        result = np.column_stack((result, col_B, col_beta))

    result = np.column_stack((result, col_lmd))

    return result

# All 1d arrays
# Pass z=None to turn off B, beta parameters
# initial_params always has 6 values: A, n, alpha, B, beta, lambda


def fitPD13Lorentzian(
        k, z, power, error, initial_params=PDW_FIT_PARAMETERS, bounds=None
):
    fitted_power = np.zeros(len(power))

    mask = np.logical_and(power > 0, error > 0)
    k_masked = k[mask]
    p_masked = power[mask]
    e_masked = error[mask]

    lb = np.full(6, -np.inf)
    ub = np.full(6, np.inf)
    lb[0] = 0
    lb[5] = 0
    ub[2] = 0  # alpha > 0 diverges at low and high k

    if z is not None:
        z_masked = z[mask]
        NUMBER_OF_PARAMS = 6
    else:
        z_masked = None
        lb[3] = 0
        lb[4] = 0
        ub[3] = 0
        ub[4] = 0
        NUMBER_OF_PARAMS = 4

    if bounds is None:
        bounds = (lb, ub)

    X_masked = (k_masked, z_masked)

    try:
        pnew, pcov = curve_fit(
            evaluatePD13Lorentz, X_masked, p_masked, initial_params,
            sigma=e_masked, absolute_sigma=True, bounds=bounds, method='trf',
            jac=jacobianPD13Lorentz)
    except ValueError:
        raise
        exit(1)
    except RuntimeError:
        raise
        exit(1)
    except OptimizeWarning:
        raise
        print("Returning initial parameters instead.")
        pnew = initial_params

    fitted_power = evaluatePD13Lorentz((k, z), *pnew)
    r = p_masked - fitted_power[mask]
    chisq = np.sum((r / e_masked)**2)
    df = len(p_masked) - NUMBER_OF_PARAMS

    fit_param_text = (
        "A        = %.3e\n"
        "n        = %.3e\n"
        "alpha    = %.3e\n"
        "B        = %.3e\n"
        "beta     = %.3e\n"
        "lambda   = %.3e\n"
    ) % (pnew[0], pnew[1], pnew[2], pnew[3], pnew[4], pnew[5])

    print(fit_param_text)
    print("chisq = %.2f," % chisq, "dof = ", df)

    return pnew, pcov


# -----------------------------------------------------
# Power spectrum ends
# -----------------------------------------------------
# -----------------------------------------------------
# Mean flux begins
# -----------------------------------------------------
BECKER13_parameters = 0.751, 2.90, -0.132, 3.5
XQ100_FIT_PARAMS = 0.89964795, 2.2378516, -0.34311581
UVES_FIT_PARAMS_WDLA = 0.4496332, 4.55802838, 0.23162296
UVES_FIT_PARAMS_NODLA = 0.41846804, 5.06996177, 0.21479074

# These are fitted to entire redshift range,
# and reported and plotted in the paper draft.
KODIAQ_MFLUX_PARAMS = 0.48554307, 4.85845246, 0.12878244
UVES_MFLUX_PARAMS = 0.46741625, 4.3688714, 0.21242962
XQ100_MFLUX_PARAMS = 2., 0.94134713, -1.45586003


@jit
def meanFluxFG08(z):
    tau = 0.001845 * np.power(1. + z, 3.924)

    return np.exp(-tau)


@jit
def evaluateBecker13MeanFlux(z, tau0, beta, C, z0=BECKER13_parameters[-1]):
    x0 = (1 + z) / (1 + z0)

    tau_eff = C + tau0 * np.power(x0, beta)

    return np.exp(-tau_eff)


def fitBecker13MeanFlux(z, F, e):
    print("z0 is fixed to {:.1f}".format(BECKER13_parameters[-1]))

    try:
        # lambda z, tau0, beta, C: evaluateBecker13MeanFlux(z, tau0, beta, C)
        pnew, pcov = curve_fit(
            evaluateBecker13MeanFlux, z, F,
            BECKER13_parameters[:-1], sigma=e, absolute_sigma=True,
            bounds=([0, 0, -2], [2, 10, 2]))
    except ValueError:
        raise
        exit(1)
    except RuntimeError:
        raise
        exit(1)
    except OptimizeWarning:
        raise
        exit(1)

    fitted_mF = evaluateBecker13MeanFlux(z, *pnew)
    r = F - fitted_mF
    chisq = np.sum((r / e)**2)
    df = len(F) - 3

    fit_param_text = (
        "tau0     = %.3e\n"
        "beta     = %.3e\n"
        "C        = %.3e\n"
    ) % (pnew[0], pnew[1], pnew[2])

    print(fit_param_text)
    print("chisq = %.2f," % chisq, "dof = ", df)

    return pnew, pcov
# -----------------------------------------------------
# Mean flux ends
# -----------------------------------------------------


def getLyaFlucErrors(
        z, dv, R_kms, lnk1=-4 * np.log(10), lnk2=-0.5 * np.log(10),
        dlnk=0.01, on_flux=True
):
    """
    Returns the VARIANCE.
    if on_flux=True, returns variance on mean flux from LSS fluctuations,
    i.e. multiplied by F-bar^2
    """
    if isinstance(z, np.ndarray):
        pass
    elif isinstance(z, float):
        z = np.array([z])
    elif isinstance(z, list):
        z = np.array(z)
    else:
        raise RuntimeError("z should be numpy array.")

    Nkpoints = int((lnk2 - lnk1) / dlnk) + 1
    k = np.exp(np.arange(Nkpoints) * dlnk + lnk1)[:, np.newaxis]

    window_fn_2 = np.sinc(k * dv / 2 / np.pi)**2 * np.exp(-k**2 * R_kms**2)

    err2_lya = np.empty(z.size)
    kv, zv = np.meshgrid(k, z, indexing='ij')
    tp2 = evaluatePD13W17Fit(kv, zv)
    tp2 *= k * window_fn_2 / np.pi

    err2_lya = np.trapz(tp2, dx=dlnk, axis=0)

    if on_flux:
        err2_lya *= meanFluxFG08(z)**2

    return err2_lya


def getLyaCorrFn(z, dlambda, log2ngrid=17, kms_grid=1., on_flux=True):
    """ Obtain RectBivariateSpline that you can call as xi1d(z, v).
    Args
    ------
    z: array_like
    Redshift values to evaluate
    dlambda: float
    Wavelength bin size

    Returns
    ------
    RectBivariateSpline
    """
    if isinstance(z, np.ndarray):
        pass
    elif isinstance(z, float):
        z = np.array([z])
    elif isinstance(z, list):
        z = np.array(z)
    else:
        raise RuntimeError("z should be numpy array.")

    if not isinstance(dlambda, float):
        raise RuntimeError("dlambda should be float.")

    ngrid = 2**log2ngrid
    v_values = kms_grid * (np.arange(ngrid) - ngrid / 2)
    xi1d = np.empty((z.size, ngrid))

    k = 2 * np.pi * np.fft.rfftfreq(ngrid, d=kms_grid)
    for zi, zr in enumerate(z):
        dv = LIGHT_SPEED * 0.8 / ((1 + zr) * LYA_WAVELENGTH)
        R_kms = dv
        window_fn_2 = np.sinc(k * dv / 2 / np.pi)**2 * np.exp(-k**2 * R_kms**2)
        tp2 = evaluatePD13W17Fit(k, zr) * window_fn_2

        if on_flux:
            tp2 *= meanFluxFG08(zr)**2

        xi1d[zi] = np.fft.fftshift(
            np.fft.irfft(tp2, n=ngrid)
        ) / kms_grid

    return RectBivariateSpline(z, v_values, xi1d, kx=1, ky=1)


# -----------------------------------------------------
# DLA begins
# -----------------------------------------------------
def equivalentWidthDLA(nhi, z_dla):
    # MBW 16.113
    w = 7.3 * np.sqrt(nhi / 10**20)  # A
    return w * (1 + z_dla)


def getNHIfromEquvalentWidthDLA(dw, z_dla):
    n = (dw / 7.3 / (1 + z_dla))**2
    return 10**20 * n


# -----------------------------------------------------
# n(z) QSO begins
# -----------------------------------------------------
def nzqso(z, N=1.17, a=-6.41, b=4.37, z0=2.2):
    x = a * (z / z0) + b * (z / z0)**2
    return N * np.exp(-x)


def cdf_zqso_unnorm(z1, z2, a=-6.41, b=4.37, z0=2.2):
    bsqrt = np.sqrt(b)
    u0 = a / 2 / bsqrt

    def _func(zp):
        u = u0 + zp * bsqrt / z0
        return erf(u)

    return _func(z2) - _func(z1)
