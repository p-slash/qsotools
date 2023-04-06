from pkg_resources import resource_filename

import fitsio
import numpy as np
from scipy.stats     import binned_statistic
from scipy.integrate import quad as scipy_quad
from scipy.stats     import norm as scipy_normal_stat
from scipy.special   import erfcinv, lambertw as scipy_lambertw
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
def a2_z(zp):
    return 58.6 * np.power((1. + zp) / 4., -2.82)

# Time evolution applied on the optical depth
def t_of_z(zp):
    return 0.55 * np.power((1. + zp) / 4., 5.1)
# Define x(z)
def x_of_z(zp, var_gauss=sigma2):
    return t_of_z(zp) * np.exp(- a2_z(zp) * var_gauss)

def Flux_d_z(delta_g, z):
    return np.exp(-x_of_z(z) * np.exp(2 * np.sqrt(a2_z(z)) * delta_g))

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

    Y = Flux_d_z(XIXI*np.sqrt(2*sigma2), ZZ)
    result = np.dot(Y, gausshermite_wi_deg25)

    return result / np.sqrt(np.pi)

def lognVarianceFluxGH(z):
    XIXI, ZZ = np.meshgrid(gausshermite_xi_deg25, z)

    Y = Flux_d_z(XIXI*np.sqrt(2*sigma2), ZZ)
    F2_mean = np.dot(Y**2, gausshermite_wi_deg25) / np.sqrt(np.pi)
    F_mean  = np.dot(Y, gausshermite_wi_deg25) / np.sqrt(np.pi)

    return F2_mean - F_mean**2 

# z is 1d array or list
def lognPowerSpGH(z, numvpoints=2**18, dv=1., corr=False):
    if isinstance(z, float):
        z = np.array([z]) 
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

        fofdeltag = lambda delta_g: np.exp(-tempxz * np.exp(delta_g))
        F1 = fofdeltag(delta1)
        F2 = fofdeltag(delta2)
        D1 = F1/mean_flux_z-1
        D2 = F2/mean_flux_z-1
        tempfunc = WW1*WW2*D1*D2

        xi_ln_f = np.transpose(np.sum(tempfunc, axis=(1,2)) / np.pi)
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

class RedshiftGenerator(object):
    """Quasar RedshiftGenerator. CDF can be read from file or 
    analytically calculated. Rescales CDF wrt zmin and zmax if 
    they are within the boundary.
    """
    def _getAnalytic(self, npoints=250):
        norm = fid.cdf_zqso_unnorm(self.zmin, self.zmax)

        self.zcdf = np.linspace(self.zmin, self.zmax, npoints)
        self.cdf  = fid.cdf_zqso_unnorm(self.zmin, self.zcdf)/norm

        return scipy_interp1d(self.cdf, self.zcdf)

    def _getFromFile(self, npoints=250):
        cdf, zcdf = np.genfromtxt(self.invcdf_file, unpack=True)
        cdf_interp = scipy_interp1d(zcdf, cdf)

        cdf1 = cdf[0]
        cdf2 = cdf[-1]

        if self.zmin > zcdf[0]:
            cdf1 = cdf_interp(self.zmin)
        else:
            self.zmin = zcdf[0]

        if self.zmax < zcdf[-1]:
            cdf2 = cdf_interp(self.zmax)
        else:
            self.zmax = zcdf[-1]

        norm = cdf2 - cdf1
        self.zcdf = np.linspace(self.zmin, self.zmax, npoints)
        self.cdf  = (cdf_interp(self.zcdf)-cdf1)/norm

        return scipy_interp1d(self.cdf, self.zcdf)

    def __init__(self, invcdf_file, zmin, zmax, use_analytic):
        self.invcdf_file = invcdf_file
        self.zmin = zmin
        self.zmax = zmax
        self.use_analytic = use_analytic

        if use_analytic:
            self.inv_cdf_interp = self._getAnalytic()
        else:
            self.inv_cdf_interp = self._getFromFile()

    def generate(self, RNST, nmocks):
        return self.inv_cdf_interp(RNST.uniform(size=nmocks))


class DLASampler():
    """ DLA sampling class.

    The DLA model is based on the column density distribution in
    fN_spline_z24.fits.gz file in pyigm, which is based on the paper cited in
    the README file. It is a smooth function, fitted to many observations. The
    redshift evolution of f(N) is (1 + z)^1.5 at pivot redshift of 2.4.
    However, we are extending well beyond the confidence interval of the said
    work.

    This class has an internal wide pixel size ``wide_pix`` in kms. It works on
    Gaussian random fields and resamples them onto this velocity spacing. This
    makes the inverse CDF easier to calculate, which is just an inverse erfc.
    When working on raw fields, DLAs were popuplated right next to each other.
    This is due to small-scale clustering. After trying different methods, I
    settled on resampling the input Gaussian field, which reduced the excess
    clustering of DLAs.

    Other methods tested:
    - Optical depth mapping on the resampled (0.2 A) skewers. The PDF turns
    out non-trivial when smoothed; and you can't control excess clustering.
    - Smoothing the Gaussian field. Without resampling, the excess clustering
    persists.

    How to use this class:
    - Initialize it at the top of your program, e.g. in main function.
    - Pass a copy of this instance to other functions and objects.
    - Then you only need to call :meth:`insert_random_dlas` to obtain a
    random DLA data array.
    """
    @staticmethod
    def convX2z(z, Om0=0.315):
        """ Conversion from X to z """
        Ez = np.sqrt((1 - Om0) + Om0 * (1 + z)**3)
        return (1 + z)**2 / Ez

    @staticmethod
    def fn_z_evo(z, zpivot=2.4, gamma_l=1.5):
        """ Redshift evolution of f(N, X)"""
        return ((1 + z) / (1 + zpivot))**gamma_l

    def __init__(
            self, wide_pix=2**10., nmin=19., nmax=23.,
            zmin=0, zmax=20., nzbins=5000
    ):
        self.nmin = nmin
        self.nmax = nmax

        self.wide_pix = None
        self.sigma_scale = None
        self.set_var_gauss(wide_pix)

        self._zbins = np.linspace(zmin, zmax, nzbins)

        fname_fn = resource_filename(
            'qsotools', 'tables/fN_spline_z24.fits.gz')

        with fitsio.FITS(fname_fn) as f:
            data = f[1].read()[0]
            self._log10N = data['LGN']
            self._fN_24 = data['FN']

        self._num_interp = None
        self.set_num_interp(nmin, nmax)

        # self._tau_c_interpolator = None
        # self._inv_cdf_interpolator = None
        # self.set_tau_c_interp(
        #     nmin, nmax, ntau_points,
        #     zbins=self._zbins)
        self.set_logn_invcdf(nmin, nmax)

    def set_var_gauss(self, wide_pix):
        lnkbins = np.linspace(-9, 5, 40000)
        kbins = np.exp(lnkbins)
        window = np.sinc(-kbins * wide_pix / 2 / np.pi)**2
        ps = lognGeneratingPower(kbins) * window * kbins
        var_gauss = np.trapz(ps, x=lnkbins) / np.pi

        self.wide_pix = wide_pix
        self.sigma_scale = np.sqrt(2 * var_gauss)

    def calc_num_systems_per_pixel(self, z, nmin=19., nmax=23.):
        w = (nmin <= self.log10N) & (self.log10N < nmax)
        lgn = self.log10N[w]
        fn = self.fN[w]

        integral_fN_at_z24 = np.trapz(10**(lgn + fn), x=lgn) * np.log(10)

        fN_z_evo = DLASampler.fn_z_evo(z)
        x2z = DLASampler.convX2z(z) * (1 + z) / fid.LIGHT_SPEED

        return  integral_fN_at_z24 * fN_z_evo * x2z * self.wide_pix

    def set_num_interp(self, nmin=19., nmax=23.):
        nums = self.calc_num_systems_per_pixel(self._zbins, nmin, nmax)
        self._num_interp = scipy_interp1d(self._zbins, nums, bounds_error=True)

    def interp_num_systems_per_pixel(self, z):
        if self._num_interp is None:
            raise Exception("Error: set_num_interp first")

        return self._num_interp(z)

    def delta_g_c(self, z):
        numsys = self.interp_num_systems_per_pixel(z)
        dgc = erfcinv(2 * numsys) * self.sigma_scale
        return dgc

    def set_logn_invcdf(self, nmin=19., nmax=23., nnhibins=400):
        lgn = np.linspace(nmin, nmax, nnhibins)
        fn = np.interp(lgn, self.log10N, self.fN)
        cdf = np.empty_like(fn)

        integrand = 10**(lgn + fn)
        dlgn = (lgn[1] - lgn[0]) * np.log(10)

        np.cumsum(integrand * dlgn, out=cdf)
        cdf -= cdf[0]
        cdf /= cdf[-1]

        self._inv_cdf_interpolator = scipy_interp1d(
            cdf, lgn, bounds_error=True)

    def set_tau_c_interp(
            self, nmin=19., nmax=23., ntau_points=1000000,
            zbins=np.linspace(1.8, 6.0, 60), dpixelA=None
    ):
        """ DEPRECATED!
        This interpolator matches CDF of tau to number of systems per
        pixel. However, it is invalid when skewers are downsampled as tau PDF
        does not hold.
        """
        if dpixelA is not None:
            self.dpixelA = dpixelA

        tau_c = np.empty_like(zbins)
        tau_data = np.empty((zbins.size, 3, ntau_points))

        for iz, z in enumerate(zbins):
            tau_edges, tau_centers, pdf = LyaMocks.pdf_tau(
                z, npoints=ntau_points)

            tau_data[iz][0] = tau_centers
            tau_data[iz][1] = pdf
            np.cumsum(pdf * np.diff(tau_edges), out=tau_data[iz][2])

        nsys = self.num_systems_per_pixel_z(zbins, nmin, nmax)

        for iz in range(zbins.size):
            itau = np.searchsorted(tau_data[iz, 2, :], 1 - nsys[iz])
            tau_c[iz] = tau_data[iz, 0, itau]

        self._tau_c_interpolator = scipy_interp1d(zbins, tau_c)

    def tau_c(self, z):
        if self._tau_c_interpolator is None:
            raise Exception("Error: set_tau_c_interp first")

        return self._tau_c_interpolator(z)

    def get_random_NHi(self, ndlas, RNST):
        u = RNST.uniform(0, 1, ndlas)
        NHi = self._inv_cdf_interpolator(u)

        return NHi

    def insert_random_dlas(self, zgrid, delta_gs, mockids, RNST):
        """ Insert random DLAs into skewers

        ``delta_gs`` are resampled onto ``wide_pix`` sized pixels before DLA
        matching.

        Args:
            zgrid (ndarray): Redshift array in the fine grid of LyaMocks
            delta_gs (ndarray): Nmocks x zgrid.size array of the Gaussian
                field of LyaMocks. Internal smoothing is applied.
            mockids (ndarray): Array of integers for MOCKID of each skewer.
            RNST (random state): Initialized random state.

        Returns:
            data_dlas (ndarray): With columns 'Z_DLA_NO_RSD', 'Z_DLA_RSD',
                'N_HI_DLA', 'MOCKID' and 'DLAID'. There is no RSD, so both
                redshifts are exactly equal. 'DLAID' is a hash of two digits
                attached to 'MOCKID'.
        """
        dvkms = fid.LIGHT_SPEED * np.log((1 + zgrid[1]) / (1 + zgrid[0]))
        m = np.round(self.wide_pix / dvkms).astype(int)
        newsize = zgrid.size // m

        def _downsample(x):
            return x[:newsize * m].reshape(newsize, m).mean(axis=1)

        refac_z = _downsample(zgrid)
        dgc = self.delta_g_c(refac_z)

        list_of_dlas = []
        total_dlas = 0
        for jj in range(mockids.size):
            w = _downsample(delta_gs[jj]) >= dgc
            num_dlas = w.sum()

            z_dlas = refac_z[w]
            Nhi_dlas = self.get_random_NHi(num_dlas, RNST)
            id_dlas = np.array([
                hash(f"{mockids[jj]:020d}{x:02d}") for x in np.arange(num_dlas)
            ], dtype=int)

            list_of_dlas.append((mockids[jj], id_dlas, z_dlas, Nhi_dlas))
            total_dlas += num_dlas

        dtype = [
            ('Z_DLA_NO_RSD', 'f8'), ('Z_DLA_RSD', 'f8'), ('N_HI_DLA', 'f8'),
            ('MOCKID', 'i8'), ('DLAID', 'i8')]

        data_dlas = np.empty(total_dlas, dtype=dtype)
        jj = 0
        for item in list_of_dlas:
            mockid, id_dlas, z_dlas, Nhi_dlas = item
            num_dlas = id_dlas.size
            data_slice = data_dlas[jj:jj + num_dlas]

            data_slice['Z_DLA_NO_RSD'] = z_dlas
            data_slice['Z_DLA_RSD'] = z_dlas
            data_slice['N_HI_DLA'] = Nhi_dlas
            data_slice['MOCKID'] = mockid
            data_slice['DLAID'] = id_dlas
            jj += num_dlas

        return data_dlas

    @property
    def log10N(self):
        return self._log10N

    @property
    def fN(self):
        return self._fN_24
        
class LyaMocks():
    """
    Generates lognormal mocks with a power spectrum similar to Lya 1D power
    spectrum up to small scales ~0.0003-0.2 s/km.

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
        resample_dv=None, obs_wave_edges=None, delta_z=None)
        Generates howmany number of mocks.
        Resamples onto the observed grid if obs_wave_edges is given.
        Also resamples onto fixed resample_dv pixel sized grid if resample_dv is given. 
        Adds gaussian noise using per kms estimate after resampling and smoothing 
        so that s/n per pixel is known.

        Cuts delta_z chunks around central redshift.
        Returns wavelength array and list of fluxes: wave, fluxes, errors.
        
    """
    @staticmethod
    def pdf_tau(
            z=3.0, tau1=-6, tau2=3.5, npoints=1000000, var_gauss=sigma2
    ):
        tau_edges = np.logspace(tau1, tau2, npoints + 1)
        tau_centers = (tau_edges[1:] + tau_edges[:-1]) / 2

        a_z = np.sqrt(a2_z(z))
        x_z = x_of_z(z, var_gauss)
        delta_g = np.log(tau_centers / x_z) / 2 / a_z

        pdf_delta_g = (
            np.exp(- delta_g**2 / var_gauss / 2)
            / np.sqrt(2 * np.pi * var_gauss)
        )
        dtau_dDeltag = 2 * a_z * tau_centers
        pdf_tau = pdf_delta_g / dtau_dDeltag

        norm = np.trapz(pdf_tau, x=tau_centers)
        pdf_tau /= norm

        return tau_edges, tau_centers, pdf_tau

    def createField(self, NMocks=1, mockids=None):
        self.delta_F = self.RNST.standard_normal((NMocks, self.N_CELLS))

        self.delta_F  = np.fft.rfft(self.delta_F, axis=1) * self.DV_KMS
        self.delta_F *= np.sqrt( self.power_spectrum_array / self.DV_KMS )

        self.delta_F = np.fft.irfft(self.delta_F, axis=1) / self.DV_KMS
        self.pick_dla_locations(mockids)

    def pick_dla_locations(self, mockids):
        """ Use it while delta_F is in k space """
        if self.dla_sampler is None:
            self.data_dlas = None
            return None

        self.data_dlas = self.dla_sampler.insert_random_dlas(
            self.z_values, self.delta_F, mockids, self.RNST)

    def __init__(
            self, SEED, N_CELLS=65536, DV_KMS=1.0, REDSHIFT_ON=True,
            GAUSSIAN_MOCKS=False, USE_LOG_V=True, dla_sampler=None
    ):
        self.RNST           = np.random.default_rng(SEED)
        self.N_CELLS        = N_CELLS
        self.DV_KMS         = DV_KMS
        self.REDSHIFT_ON    = REDSHIFT_ON
        self.GAUSSIAN_MOCKS = GAUSSIAN_MOCKS
        self.dla_sampler = dla_sampler

        v_values  = self.DV_KMS * (np.arange(self.N_CELLS) - self.N_CELLS/2)
        self.k_values  = 2. * np.pi * np.fft.rfftfreq(self.N_CELLS, d=self.DV_KMS)

        if USE_LOG_V:
            self.z0_values = np.exp(v_values / fid.LIGHT_SPEED) - 1
        else:
            self.z0_values = (1 - v_values / 2. / fid.LIGHT_SPEED)**-2 - 1

        self.Z_CENTER = 0
        
        if self.GAUSSIAN_MOCKS:
            self.power_spectrum_array = fid.evaluatePD13W17Fit(self.k_values)
        else:
            self.power_spectrum_array = lognGeneratingPower(self.k_values)

        self.init_variance = np.trapz(
            self.power_spectrum_array, dx=self.k_values[1]) / np.pi

    def setCentralRedshift(self, Z_C):
        self.Z_CENTER = Z_C
        self.z_values = (1 + self.Z_CENTER) * (1 + self.z0_values) - 1

        if self.REDSHIFT_ON:
            self.evo_redshifts = self.z_values
        else:
            self.evo_redshifts = np.ones(self.N_CELLS) * self.Z_CENTER

    def smoothGaussian(self, R):
        if R is not None:
            k = self.k_values
            delta_k  = np.fft.rfft(self.delta_F, axis=1) * self.DV_KMS
            delta_k *= np.exp(-k*k * R*R / 2.0)

            self.delta_F = np.fft.irfft(delta_k, axis=1) / self.DV_KMS

    def applySpectographResolution(self, spectrograph_resolution):
        if spectrograph_resolution:
            self.smoothGaussian(
                fid.LIGHT_SPEED / spectrograph_resolution
                / fid.ONE_SIGMA_2_FWHM)

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

    def generateMocks(
        self, NMocks=1, spectrograph_resolution=None, R_tau=None, mockids=None
    ):
        self.createField(NMocks, mockids)

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

    def resampledMocks(
            self, howmany, err_per_final_pixel=0, spectrograph_resolution=None,
            resample_dv=None, obs_wave_edges=None, delta_z=None,
            keep_empty_bins=False, mockids=None
    ):
        wave = fid.LYA_WAVELENGTH * (1. + self.z_values)

        self.generateMocks(howmany, spectrograph_resolution, mockids=mockids)
        fluxes = self.delta_F

        # Resample onto an observed grid if obs_wave_centers given
        if obs_wave_edges is not None:
            # Create wavelength edges for logarithmicly spaced array from obs_wave_centers
            # or linearly spaced array if logspacing_obswave=False
            # obs_wave_edges = specops.createEdgesFromCenters(obs_wave_centers, logspacing=logspacing_obswave)
            wave, fluxes = specops.resample(wave, fluxes, None, obs_wave_edges, keep_empty_bins)

            # Break if observed grid does not include the wavelength range
            if len(wave) == 0:
                return wave, 0, 0

        if resample_dv:
            wave, fluxes = specops.resample(wave, fluxes, None, resample_dv, keep_empty_bins)

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
