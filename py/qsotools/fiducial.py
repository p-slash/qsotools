import numpy as np
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.integrate import trapz as scipy_trapz
from scipy.signal import fftconvolve
from scipy.linalg import sqrtm as sla_sqrtm
from scipy.special import erf

warnings.simplefilter("error", OptimizeWarning)

LYA_WAVELENGTH  = 1215.67
LYA_FIRST_WVL   = 1050.
LYA_LAST_WVL    = 1180.
LYA_CENTER_WVL  = (LYA_LAST_WVL + LYA_LAST_WVL) / 2

Si4_FIRST_WVL    = 1268.
Si4_LAST_WVL     = 1380.

C4_FIRST_WVL     = 1409.
C4_LAST_WVL      = 1523.

LIGHT_SPEED      = 299792.458
ONE_SIGMA_2_FWHM = 2.35482004503

def formBins(nblin, nblog, dklin, dklog, k0, klast=-1):
    lin_bin_edges = np.arange(nblin+1) * dklin + k0
    log_bin_edges = lin_bin_edges[-1] * np.power(10., np.arange(1, nblog + 1) * dklog)
    
    # assert log_bin_edges[-1] < k_values[-1]

    bin_edges   = np.concatenate((lin_bin_edges, log_bin_edges))
    if klast > bin_edges[-1]:
        bin_edges.append(klast)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    return bin_edges, bin_centers

# Assuming R is integer resolution power
# dv in km/s and k in s/km
def getSpectographWindow_k(k, Rint, dv):
    Rv = LIGHT_SPEED / Rint / ONE_SIGMA_2_FWHM
    x = k*dv/2/np.pi # numpy sinc convention multiplies x with pi
    
    W2k = np.exp(-(k*Rv)**2/2) * np.sinc(x)
    return W2k

def getSpectographWindow_x(x, Rint, dv):
    Rv = LIGHT_SPEED / Rint / ONE_SIGMA_2_FWHM
    gamma_p = (x + (dv/2))/Rv/np.sqrt(2)
    gamma_m = (x - (dv/2))/Rv/np.sqrt(2)
    
    return (erf(gamma_p)-erf(gamma_m))/2/dv

def getGaussianResolutionMatrix(Ngrid, Rint, dv, ndiags=11):
    resomat = np.empty((ndiags, Ngrid))
    offsets = np.arange(ndiags//2,-(ndiags//2)-1,-1)

    for i in range(Ngrid):
        resomat[:, i]=getSpectographWindow_x(offsets*dv, Rint, dv)*dv

    return resomat

def getOptimalResolutionMatrix(Ngrid, xi, Rint, dv, ndiags=11):
    # Calculate true correlation function
    v_xi = np.arange(xi.size)
    v_xi -= v_xi[v_xi.size//2]

    window = getSpectographWindow_x(v_xi, Rint, dv)

    # fftconvolve shifts the array by one index for some reason
    xi_2 = fftconvolve(xi, window, mode='same')
    xi_2 = np.roll(xi_2, (np.argmax(xi)-np.argmax(xi_2)))
    xi_2 = fftconvolve(xi_2, window, mode='same')
    xi_2 = np.roll(xi_2, (np.argmax(xi)-np.argmax(xi_2)))

    grid = (np.arange(Ngrid)-Ngrid/2)*dv
    S = np.empty((Ngrid, Ngrid))
    S2 = np.empty((Ngrid, Ngrid))
    for i in range(Ngrid):
        vtemp = grid - grid[i]
        S[:, i] = np.interp(vtemp, v_xi, xi)
        S2[:, i] = np.interp(vtemp, v_xi, xi_2)

    Ropt = sla_sqrtm(S2)@np.linalg.inv(sla_sqrtm(S))

    # compress
    offsets = np.arange(ndiags//2,-(ndiags//2)-1,-1)
    resomat = np.zeros((ndiags, Ngrid))
    # offsets: [ 5  4  3  2  1  0 -1 -2 -3 -4 -5]
    # when offsets[i]>0, remove initial offsets[i] elements from resomat.T[i]
    # when offsets[i]<0, remove last |offsets[i]| elements from resomat.T[i]
    for i in range(ndiags):
        off = offsets[i]
        if off>=0:
            resomat[i, off:] = Ropt.diagonal(off)
        else:
            resomat[i, :off] = Ropt.diagonal(off)

    return resomat

# -----------------------------------------------------
# Power spectrum begins
# -----------------------------------------------------
PDW_FIT_AMP  = 6.62141965e-02
PDW_FIT_N    = -2.68534876e+00
PDW_FIT_APH  = -2.23276251e-01
PDW_FIT_B    = 3.59124427e+00
PDW_FIT_BETA = -1.76804541e-01
PDW_FIT_LMD  = 3.59826056e+02

PDW_FIT_PARAMETERS       = PDW_FIT_AMP, PDW_FIT_N, PDW_FIT_APH, PDW_FIT_B, PDW_FIT_BETA, PDW_FIT_LMD
PDW_FIT_PARAMETERS_0BETA = PDW_FIT_AMP, PDW_FIT_N, PDW_FIT_APH, PDW_FIT_B, 0, PDW_FIT_LMD
PD13_PIVOT_K = 0.009
PD13_PIVOT_Z = 3.0

def evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd):
    k, z = X
    q0 = k/PD13_PIVOT_K + 1e-10

    result = (A*np.pi/PD13_PIVOT_K) * np.power(q0, 2. + n + alpha*np.log(q0)) / (1. + lmd * k**2)
    
    if z is not None:
        x0 = (1. + z) / (1. + PD13_PIVOT_Z)
        result *= np.power(q0, beta * np.log(x0)) * np.power(x0, B)
    
    return result

def evaluatePD13W17Fit(k, z=None):
    if z is None:
        p_noredshif     = list(PDW_FIT_PARAMETERS)
        p_noredshif[3]  = 0
        p_noredshif[4]  = 0
        
        pfid    = p_noredshif
        z       = 3.0
    else:
        pfid    = list(PDW_FIT_PARAMETERS_0BETA)

    return evaluatePD13Lorentz((k, z), *pfid)

def jacobianPD13Lorentz(X, A, n, alpha, B, beta, lmd):
    k, z = X
    pkz = evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd)
    lnk = np.log(k / PD13_PIVOT_K + 1e-10)
    
    col_A     = pkz / A
    col_n     = pkz * lnk
    col_alpha = pkz * lnk * lnk
    col_lmd   = -pkz * k**2 / (1. + lmd * k**2)
    
    result = np.column_stack((col_A, col_n, col_alpha))

    if z is not None:
        lnz = np.log((1. + z) / (1. + PD13_PIVOT_Z))
        col_B     = pkz * lnz
        col_beta  = pkz * lnk * lnz
        result = np.column_stack((result, col_B, col_beta))

    result = np.column_stack((result, col_lmd))

    return result

# All 1d arrays
# Pass z=None to turn off B, beta parameters
# initial_params always has 6 values: A, n, alpha, B, beta, lambda
def fitPD13Lorentzian(k, z, power, error, initial_params=PDW_FIT_PARAMETERS, bounds=None):
    fitted_power = np.zeros(len(power))

    mask     = np.logical_and(power > 0, error > 0)
    k_masked = k[mask]
    p_masked = power[mask]
    e_masked = error[mask]

    lb = np.full(6, -np.inf)
    ub = np.full(6, np.inf)
    lb[0] = 0
    lb[5] = 0
    ub[2] = 0 # alpha > 0 diverges at low and high k

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
        pnew, pcov = curve_fit(evaluatePD13Lorentz, X_masked, p_masked, initial_params, \
            sigma=e_masked,  absolute_sigma=True, bounds=bounds, method='trf', \
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

    fitted_power = evaluatePD13Lorentz((k,z), *pnew)
    r     = p_masked - fitted_power[mask]
    chisq = np.sum((r/e_masked)**2)
    df    = len(p_masked) - NUMBER_OF_PARAMS

    fit_param_text = ("A        = %.3e\n" "n        = %.3e\n" "alpha    = %.3e\n"
        "B        = %.3e\n" "beta     = %.3e\n" "lambda   = %.3e\n") % (
        pnew[0], pnew[1], pnew[2], pnew[3], pnew[4], pnew[5])

    print(fit_param_text)
    print("chisq = %.2f,"%chisq, "dof = ", df)
    
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
UVES_MFLUX_PARAMS   = 0.46741625, 4.3688714,  0.21242962
XQ100_MFLUX_PARAMS  = 2.        , 0.94134713, -1.45586003

def meanFluxFG08(z):
    tau = 0.001845 * np.power(1. + z, 3.924)

    return np.exp(-tau)

def evaluateBecker13MeanFlux(z, tau0, beta, C, z0=BECKER13_parameters[-1]):
    x0 = (1+z) / (1+z0)

    tau_eff = C + tau0 * np.power(x0, beta)

    return np.exp(-tau_eff)

def fitBecker13MeanFlux(z, F, e):
    print("z0 is fixed to {:.1f}".format(BECKER13_parameters[-1]))

    try:
        # lambda z, tau0, beta, C: evaluateBecker13MeanFlux(z, tau0, beta, C)
        pnew, pcov = curve_fit(evaluateBecker13MeanFlux, \
            z, F, BECKER13_parameters[:-1], sigma=e, absolute_sigma=True, \
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
    r     = F - fitted_mF
    chisq = np.sum((r/e)**2)
    df    = len(F) - 3

    fit_param_text = ("tau0     = %.3e\n" "beta     = %.3e\n" "C        = %.3e\n") % (
        pnew[0], pnew[1], pnew[2])

    print(fit_param_text)
    print("chisq = %.2f,"%chisq, "dof = ", df)

    return pnew, pcov

# -----------------------------------------------------
# Mean flux ends
# -----------------------------------------------------

"""
return variance on mean flux from LSS fluctuations, i.e. multiplied by F-bar^2
"""
def getLyaFlucErrors(z, dv, R_kms, logk1=-6, logk2=1, npoints=1000):
    window_fn = lambda k: np.sinc(k*dv/2/np.pi) * np.exp(-k**2 * R_kms**2/2)
    kPpi      = lambda k, z1: k * evaluatePD13W17Fit(k,z1) / np.pi

    flnk = lambda lnk, z1: kPpi(np.exp(lnk), z1) * window_fn(np.exp(lnk))**2

    klog = np.linspace(logk1*np.log(10), logk2*np.log(10), npoints)
    ZZ, KK = np.meshgrid(z, klog, indexing='ij')
    
    err2_lya = scipy_trapz(flnk(KK, ZZ), KK) * meanFluxFG08(z)**2

    return err2_lya

# -----------------------------------------------------
# DLA begins
# -----------------------------------------------------    

# MBW 16.113
def equivalentWidthDLA(nhi, z_dla):
    w = 7.3 * np.sqrt(nhi/10**20) # A
    return w*(1+z_dla)

def getNHIfromEquvalentWidthDLA(dw, z_dla):
    n = (dw/7.3/(1+z_dla))**2
    return 10**20 * n







