import numpy as np
from scipy.stats     import binned_statistic
from scipy.integrate import quad as scipy_quad
from scipy.stats     import norm as scipy_normal_stat
from scipy.special   import lambertw as scipy_lambertw
from scipy.interpolate import interp1d as scipy_interp1d

from qsotools import specops
import qsotools.fiducial as fid

# Power spectrum functions
# ----------------------------
def lognGeneratingPower(k):
    n     = 0.5
    alpha = 0.26
    gamma = 1.8

    k_0 = 0.001
    k_1 = 0.04

    q0 = k / k_0 + 1e-10

    result = np.power(q0, n - alpha * np.log(q0)) / (1. + np.power(k/k_1, gamma))

    return result

def xi_g_v_fn(v):
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype=np.double)

    result = np.zeros_like(v)
    
    for i, vv in enumerate(v):
        result[i] = scipy_quad(lognGeneratingPower, 0, np.inf, weight='cos', wvar=vv)[0] / np.pi
    
    return result

# ----------------------------

# Variance of the Gaussian field
sigma2 = scipy_quad(lognGeneratingPower, 0, np.inf, limit=100000)[0]/np.pi

# Time evolution applied on the Gaussian field
a2_z   = lambda zp: 58.6 * np.power((1. + zp) / 4., -2.82)

# Time evolution applied on the optical depth
t_of_z = lambda zp: 0.55 * np.power((1. + zp) / 4., 5.1)
# Define x(z)
x_of_z = lambda zp: t_of_z(zp) * np.exp(- a2_z(zp) * sigma2)

def lognMeanFluxSaddle(z):
    sigma2z = a2_z(z) * sigma2
    tempxz = t_of_z(z) * np.exp(-sigma2z)

    d_s = - scipy_lambertw(4*sigma2z*tempxz).real / 2
    
    return np.exp(d_s * (1-d_s) / 2 / sigma2z) / np.sqrt(1 - 2*d_s)

def lognMeanFlux0(z):
    tempxz = x_of_z(z)
    y_of_z = 4 * a2_z(z) * sigma2 * tempxz

    return np.exp(- tempxz * (2 + y_of_z) / (2 + 2*y_of_z)) / np.sqrt(1 + y_of_z)

# Precompute Gauss-Hermite numbers
gausshermite_xi_deg25, gausshermite_wi_deg25 = np.polynomial.hermite.hermgauss(25)
# Assume z is 1d array
def lognMeanFluxGH(z):
    XIXI, ZZ = np.meshgrid(gausshermite_xi_deg25, z)
    sigmaz = np.sqrt(a2_z(ZZ) * sigma2)
    tempxz = x_of_z(ZZ)
    
    result = np.dot(np.exp(-tempxz * np.exp(2 * np.sqrt(2) * sigmaz * XIXI)), gausshermite_wi_deg25)
    
    return result / np.sqrt(np.pi)

# z is 1d array or list
def lognPowerSpGH(z, numvpoints=2**18, dv=1., corr=False):    
    # Set up k array
    k_arr  = 2. * np.pi * np.fft.rfftfreq(numvpoints, d=dv)

    # Compute correlation function for the underlying Gaussian fiels
    xi_gaus_v = np.fft.irfft(lognGeneratingPower(k_arr)).real / dv #[:int(numvpoints/2)]
    sigma2    = xi_gaus_v[0]
    xi_sine   = xi_gaus_v/sigma2
    xi_sine[xi_sine>1] = 1.
    xi_cosine = np.sqrt(1 - xi_sine**2)
    XI_VEC    = np.array([xi_sine, xi_cosine]).transpose()

    # Set up Gauss-Hermite quadrature
    YY1, YY2 = np.meshgrid(gausshermite_xi_deg25, gausshermite_xi_deg25, indexing='ij')
    WW1, WW2 = np.meshgrid(gausshermite_wi_deg25, gausshermite_wi_deg25, indexing='ij')
    YY2_XI_VEC_WEIGHTED = np.dot(XI_VEC, np.array([YY1, YY2]).transpose(1,0,2))
    
    if corr:
        corr_results_arr = np.zeros((z.size, numvpoints))
    else:
        power_results_arr = np.zeros((z.size, k_arr.size))
    # Redshift dependent 
    for zi, zr in enumerate(z):
        mean_flux_z = lognMeanFluxGH(zr)
        sigmaz = np.sqrt(a2_z(zr) * sigma2)
        tempxz = x_of_z(zr)
        delta1 = YY1 * sigmaz * 2 * np.sqrt(2)
        delta2 = YY2_XI_VEC_WEIGHTED * sigmaz * 2 * np.sqrt(2)

        tempfunc = WW1*WW2*np.exp(-tempxz * (np.exp(delta1) + np.exp(delta2)))
        
        xi_ln_f = np.transpose(np.sum(tempfunc, axis=(1,2)) / np.pi / mean_flux_z**2 - 1)
        if corr:
            corr_results_arr[zi] = xi_ln_f
        else:
            power_results_arr[zi] = np.fft.rfft(xi_ln_f).real * dv
    
    if corr:
        return np.arange(numvpoints) * dv, corr_results_arr
    else:
        return k_arr, power_results_arr

def lognNoisePowerSp(err_flux, z, dv):
    rs = err_flux / lognMeanFluxGH(z) * dv
    return rs**2

def genContinuumError(wave, se0, se1):
    lnwave = np.log(wave)
    slope  = (lnwave - lnwave[0]) / (lnwave[-1] - lnwave[0])

    return se0 + se1 * slope

class LyaMocks():
    """
    Generates lognormal mocks with a power spectrum similar to Lya 1D power spectrum 
    up to small scales ~0.0003-0.2 s/km.

    Parameters
    ----------
    SEED : int
        Seed to generate random numbers.
    N_CELLS : int
        Number of pixels or cells. Default is 2**16 = 65536.
    DV_KMS  : float
        Pixel width. Default is 1.0.
    REDSHIFT_ON : Bool
        Turn on redshift evolution. Default is True.
    GAUSSIAN_MOCKS : Bool
        Generate Gaussian mocks instead of log-normal mocks. 
        Default is False, i.e. generating log-normal mocks.

    __init__(SEED, N_CELLS=65536, DV_KMS=1.0, REDSHIFT_ON=True, GAUSSIAN_MOCKS=False, USE_LOG_V=True)
        Creates a grid, equal spacing in velocity DV_KMS. Computes redshift values, 
        the power spectrum for this array.

    Attributes
    ----------
    RNST : RandomState.
        State of random generator with given SEED.
    N_CELLS : int
        Number of pixels or cells.
    DV_KMS  : float
        Pixel width.
    REDSHIFT_ON : Bool
        Turn on redshift evolution. Default is True.
    GAUSSIAN_MOCKS : Bool
        Generate Gaussian mocks instead of log-normal mocks. Default is False, 
        i.e. generating log-normal mocks.
    Z_CENTER : float
        Central redshift of the grid. Set by setCentralRedshift(Z_C).

    v_values : list
    z0_values : list
    z_values : list
    k_values : list
    evo_redshifts : list
        Hold velocity, redshift and k in Fourier space values respectively. 
        evo_redshifts holds Z_C if REDSHIFT_ON=False, z_values if True.

    delta_F : list
        The grid that eventually becomes Flux.
    power_spectrum_array : list
        Generating power spectrum is precomputed and saved on the grid.
    init_variance : float
        Variance of the initial Gaussian random field. Computed by summing up the power spectrum array.

    Methods
    -------
    createField(NMocks=1)
        Generate a Gaussian random field g(z) with power spectrum given by 
        lognGeneratingPower or fiducial.evaluatePD13W17Fit for Gaussian mocks.
        Stored in delta_F.
    setCentralRedshift(Z_C)
        Set Z_C as central redshift for the long grid (Z_CENTER). Sets up z_values, redshifts, 
        power_spectrum_array and init_variance.
    lognTransform(R_tau=None)
        Multiply the Gaussian field g(z) by a(z)=z_evolution_of_power_spectrum_sqrt(z).
        Then transforms by 
            n(z) = exp(2a(z)g(z) - a(z)**2 init_variance)
        Smoothes with a Gaussian kernel of radius R_tau if passed.
        Stored in delta_F.
    smoothGaussian(R)
        Smooth delta_F with a Gaussian kernel using FFT.
    applySpectographResolution(self, spectrograph_resolution)
        Applies spectrograph resolution as a Gaussian smoothing to delta_F. 
        Takes integer FWHM resolution.
    transformTauField()
        Transforms by tau(z) = n(z) * z_evolution_of_tau(z)
        Stored in delta_F.
    transformFlux()
        Transforms by F(z) = exp(-tau(z)). 
        Stored in delta_F.

    generateMocks(NMocks=1, spectrograph_resolution=None, R_tau=None)
        Generates NMocks mocks with R_tau in lognTransform and smoothes the flux 
        with spectrograph resolution. 
        Does not resample onto another grid.
    
    resampledMocks(self, howmany, err_per_final_pixel=0, spectrograph_resolution=None, \
        resample_dv=None, obs_wave_centers=None, delta_z=None)
        Generates howmany number of mocks.
        Resamples onto the observed grid if obs_wave_centers is given.
        Also resamples onto fixed resample_dv pixel sized grid if resample_dv is given. 
        Adds gaussian noise using per kms estimate after resampling and smoothing 
        so that s/n per pixel is known.

        Cuts delta_z chunks around central redshift.
        Returns wavelength array and list of fluxes: wave, fluxes, errors.
        
    """
    def createField(self, NMocks=1):
        self.delta_F = self.RNST.standard_normal((NMocks, self.N_CELLS))

        delta_k  = np.fft.rfft(self.delta_F, axis=1) * self.DV_KMS
        delta_k *= np.sqrt( self.power_spectrum_array / self.DV_KMS )
        
        self.delta_F = np.fft.irfft(delta_k, axis=1) / self.DV_KMS

    def __init__(self, SEED, N_CELLS=65536, DV_KMS=1.0, REDSHIFT_ON=True, \
        GAUSSIAN_MOCKS=False, USE_LOG_V=True):
        self.RNST           = np.random.RandomState(SEED)
        self.N_CELLS        = N_CELLS
        self.DV_KMS         = DV_KMS
        self.REDSHIFT_ON    = REDSHIFT_ON
        self.GAUSSIAN_MOCKS = GAUSSIAN_MOCKS

        self.v_values  = self.DV_KMS * (np.arange(self.N_CELLS) - self.N_CELLS/2)
        self.k_values  = 2. * np.pi * np.fft.rfftfreq(self.N_CELLS, d=self.DV_KMS)

        if USE_LOG_V:
            self.z0_values = np.exp(self.v_values / fid.LIGHT_SPEED) - 1.
        else:
            self.z0_values = np.power(1. - self.v_values / 2. / fid.LIGHT_SPEED, -2) - 1.
        
        self.Z_CENTER = 0
        
        if self.GAUSSIAN_MOCKS:
            self.power_spectrum_array = fid.evaluatePD13W17Fit(self.k_values)
        else:
            self.power_spectrum_array = lognGeneratingPower(self.k_values)

        self.init_variance = np.sum(self.power_spectrum_array) * self.k_values[1] / np.pi

    def setCentralRedshift(self, Z_C):
        self.Z_CENTER = Z_C
        self.z_values = (1 + self.Z_CENTER) * (1 + self.z0_values) - 1
                
        if self.REDSHIFT_ON:
            self.evo_redshifts = self.z_values
        else:
            self.evo_redshifts = np.ones(self.N_CELLS) * self.Z_CENTER

    def smoothGaussian(self, R):
        if R:
            k = self.k_values
            delta_k  = np.fft.rfft(self.delta_F, axis=1) * self.DV_KMS
            delta_k *= np.exp(-k*k * R*R / 2.0)

            self.delta_F = np.fft.irfft(delta_k, axis=1) / self.DV_KMS

    def applySpectographResolution(self, spectrograph_resolution):
        if spectrograph_resolution:
            self.smoothGaussian(fid.LIGHT_SPEED / spectrograph_resolution / fid.ONE_SIGMA_2_FWHM)

    def redshiftEvolutionGaussian(self):
        a_z = np.power((1 + self.evo_redshifts) / 4, fid.PDW_FIT_B / 2)
        self.delta_F *= a_z

    def lognTransform(self, R_tau=None):
        a2_z = 58.6 * np.power((1 + self.evo_redshifts) / 4, -2.82)

        self.delta_F *= np.sqrt(a2_z)
        variance      = a2_z * self.init_variance

        self.delta_F = np.exp(2 * self.delta_F - variance)

        self.smoothGaussian(R_tau)

    def transformTauField(self):
        t_z = 0.55 * np.power((1 + self.evo_redshifts) / 4, 5.1)

        self.delta_F *= t_z

    def transformFlux(self):
        self.delta_F = np.exp(-self.delta_F)

    def generateMocks(self, NMocks=1, spectrograph_resolution=None, R_tau=None):
        self.createField(NMocks)
        
        if self.GAUSSIAN_MOCKS:
            # The fiducial power spectrum has NOT redshift evolution of fluctuations in it.
            # Multiply by (1.+z / 4)^B/2
            self.redshiftEvolutionGaussian()

            # Now add redshift evolution of the mean flux.
            self.delta_F = fid.meanFluxFG08(self.evo_redshifts) * (1 + self.delta_F)
        else:
            self.lognTransform(R_tau)
            self.transformTauField()
            self.transformFlux()

        # if add_err:
        #     # std_err = 0.1 # Average noise in KODIAQ is 0.1, see KFITS.py
        #     self.delta_F += self.generateGaussianNoise(add_err, self.delta_F)

        # Resampling reduces noise by square root. However, spectrograph smoothing
        # reduces noise in non-trivial way. Mathematically, adding noise before or after
        # should be equal anyway.
        self.applySpectographResolution(spectrograph_resolution)
    
    def generateGaussianNoise(self, std_err, arr):
        return self.RNST.normal(loc=0, scale=std_err, size=arr.shape)

    def qsoMock(self, qso, spectrograph_resolution=None, const_error=None):
        wave = fid.LYA_WAVELENGTH * (1. + self.z_values)

        self.generateMocks(spectrograph_resolution=spectrograph_resolution)
        qso.error = qso.error.reshape(1, qso.error.size)
        
        # Resample everything back to observed grid 
        # Create wavelength edges for logarithmicly spaced array from obs_wave_centers
        obs_wave_edges = specops.createEdgesFromCenters(qso.wave)
        wave, qso.flux = specops.resample(wave, self.delta_F, None, obs_wave_edges)
        obs_wave_edges = specops.createEdgesFromCenters(wave)
        qso.wave, qso.error = specops.resample(qso.wave, qso.error, None, obs_wave_edges)
        qso.size = len(qso.wave)

        # Break if observed grid does not include the wavelength range
        if qso.size == 0:
            raise ValueError("Empty grid")

        if const_error:
            qso.error = const_error * np.ones_like(qso.flux)

        qso.flux += self.generateGaussianNoise(qso.error, qso.flux)

        qso.flux = qso.flux.ravel()
        qso.error = qso.error.ravel()
        qso.specres = spectrograph_resolution

        return qso

    def resampledMocks(self, howmany, err_per_final_pixel=0, spectrograph_resolution=None, \
        resample_dv=None, obs_wave_centers=None, delta_z=None, logspacing_obswave=True):
        wave   = fid.LYA_WAVELENGTH * (1. + self.z_values)

        self.generateMocks(howmany, spectrograph_resolution)
        fluxes = self.delta_F

        # Resample onto an observed grid if obs_wave_centers given
        if obs_wave_centers is not None:
            # Create wavelength edges for logarithmicly spaced array from obs_wave_centers
            # or linearly spaced array if logspacing_obswave=False
            obs_wave_edges = specops.createEdgesFromCenters(obs_wave_centers, logspacing=logspacing_obswave)
            wave, fluxes = specops.resample(wave, fluxes, None, obs_wave_edges)

            # Break if observed grid does not include the wavelength range
            if len(wave) == 0:
                return wave, 0, 0

        if resample_dv:
            wave, fluxes = specops.resample(wave, fluxes, None, resample_dv)

        # Add Gaussian noise on every pixel with respect to its error
        if err_per_final_pixel:
            errors = err_per_final_pixel * np.ones_like(fluxes)
            fluxes += self.generateGaussianNoise(errors, fluxes)
        else:
            errors = np.zeros_like(fluxes)

        if delta_z:
            wave_max = fid.LYA_WAVELENGTH * (1. + self.Z_CENTER + delta_z/2.)
            wave_min = fid.LYA_WAVELENGTH * (1. + self.Z_CENTER - delta_z/2.)

            fluxes = np.array([f[(wave_min < wave) & (wave < wave_max)] for f in fluxes])
            errors = np.array([e[(wave_min < wave) & (wave < wave_max)] for e in errors])
            wave   = wave[(wave_min < wave) & (wave < wave_max)]

        return wave, fluxes, errors


            









































