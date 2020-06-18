import numpy as np
import warnings
from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)

LYA_WAVELENGTH  = 1215.67
LYA_FIRST_WVL   = 1050.
LYA_LAST_WVL    = 1180.
LYA_CENTER_WVL  = (LYA_LAST_WVL + LYA_LAST_WVL) / 2

LIGHT_SPEED      = 299792.458
ONE_SIGMA_2_FWHM = 2.35482004503

# -----------------------------------------------------
# Power spectrum begins
# -----------------------------------------------------
#                                 A               n                alpha            B               beta             lmd
PD13_fiducial_parameters        = 6.62141965e-02, -2.68534876e+00, -2.23276251e-01, 3.59124427e+00, -1.76804541e-01, 3.59826056e+02
PD13_fiducial_parameters_0beta  = 6.62141965e-02, -2.68534876e+00, -2.23276251e-01, 3.59124427e+00, 0,               3.59826056e+02
PD13_PIVOT_K = 0.009
PD13_PIVOT_Z = 3.0

def evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd):
    k, z = X
    q0 = k / PD13_PIVOT_K + 1e-10

    result = (A * np.pi / PD13_PIVOT_K) * np.power(q0, 2. + n + alpha * np.log(q0)) / (1. + lmd * k**2)
    
    if z is not None:
        x0 = (1. + z) / (1. + PD13_PIVOT_Z)
        result *= np.power(q0, beta * np.log(x0)) * np.power(x0, B)
    
    return result

def evaluatePD13W17Fit(k, z=None):
    if z is None:
        p_noredshif     = list(PD13_fiducial_parameters)
        p_noredshif[3]  = 0
        p_noredshif[4]  = 0
        
        pfid    = p_noredshif
        z       = 3.0
    else:
        pfid    = list(PD13_fiducial_parameters_0beta)

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
def fitPD13Lorentzian(k, z, power, error, initial_params=PD13_fiducial_parameters):
    fitted_power = np.zeros(len(power))

    mask     = np.logical_and(power > 0, error > 0)
    k_masked = k[mask]
    p_masked = power[mask]
    e_masked = error[mask]

    lb = np.full(6, -np.inf)
    ub = np.full(6, np.inf)
    lb[0] = 0
    lb[5] = 0

    if z is not None:
        z_masked = z[mask]
        NUMBER_OF_PARAMS = 6
    else:
        z_masked = None
        lb[3] = 0
        lb[4] = 0
        ub[3] = 0
        ub[4] = 0
        NUMBER_OF_PARAMS =4 

    X_masked = (k_masked, z_masked)

    try:
        pnew, pcov = curve_fit(evaluatePD13Lorentz, X_masked, p_masked, initial_params, sigma=e_masked, \
            absolute_sigma=True, bounds=(lb, ub), method='trf', jac=jacobianPD13Lorentz)
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
        pnew, pcov = curve_fit(evaluateBecker13MeanFlux, z, F, BECKER13_parameters, sigma=e, \
            absolute_sigma=True)
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

    








