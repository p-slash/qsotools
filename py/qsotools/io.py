import struct
from configparser import ConfigParser
from os.path import exists as os_exists, join as ospath_join

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import zscore as scipy_zscore, norm as scipy_norm
from scipy.ndimage import median_filter as scipy_median_filter

import fitsio

from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.units import hourangle, deg

from qsotools.fiducial import LIGHT_SPEED, LYA_WAVELENGTH, \
    LYA_FIRST_WVL, LYA_LAST_WVL, formBins, equivalentWidthDLA

from pkg_resources import resource_filename
TABLE_KODIAQ_ASU    = resource_filename('qsotools', 'tables/kodiaq_asu.tsv')
TABLE_KODIAQ_MASTER = resource_filename('qsotools', 'tables/master_kodiaq_table.tsv')
TABLE_XQ100_SUM     = resource_filename('qsotools', 'tables/xq100_thework.fits')
TABLE_XQ100_DLA     = resource_filename('qsotools', 'tables/xq100_dla_table_sanchez-ramirez_2016.csv')
TABLE_SQUAD_DR1     = resource_filename('qsotools', 'tables/uves_squad_dr1_quasars_master.csv')

class Spectrum:
    """
    A generic spectrum object. Sets up a mask where error > 0.

    Parameters
    ----------
    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.    
    z_qso : float
        Emission redshift of the quasar.
    specres : int
        Spectral resolution of the instrument.
    dv : float
        Pixel width.
    ra : float
        Right ascension in radians
    dec : float
        Declination in radians

    __init__(self, wave, flux, error, z_qso, specres, dv, ra, dec)
        Creates this spectrum object. Computes s2n.

    Attributes
    ----------
    mask : 
        Good pixels on full spectrum, error>0 by default.
    size : int
        Length of arrays.
    s2n : float
        Signal to noise ratio of the entire spectrum as ave(1/error).
    s2n_lya : float
        Signal to noise ratio of the Lya forest. -1 if there is no Lya coverage for a given spectrum.
    z_dlas : list, float
        Redshift of DLAs. Needs to be set by hand. None as default
    nhi_dlas : list, float
        Column density of DLAs. Needs to be set by hand. None as default

    Methods
    -------
    applyMask(good_pixels=None)
        Remove masked values from wave, flux and error. 
        Keeps good_pixels and updates the length the arrays.

    maskZScore(thres=3.5)
        Mask pixels (flux and error) by their zscore, z=|x-mean|/std.

    getS2NLya(lya_lower=fid.LYA_FIRST_WVL, lya_upper=fid.LYA_LAST_WVL)
        Computes the signal-to-noise in lyman alpha region as average(1/e)

    """
    def __init__(self, wave, flux, error, z_qso, specres, dv, ra, dec):
        self.wave  = wave
        self.flux  = flux
        self.error = error
        self.z_qso = z_qso
        self.specres = specres
        self.dv = dv
        self.ra = ra
        self.dec = dec
        
        self.size = len(self.wave)
        self.mask = error > 0
        self.s2n = np.mean(1./error[self.mask])
        self.s2n_lya = self.getS2NLya()

        self.z_dlas = None
        self.nhi_dlas = None

    def cutForestAnalysisRegion(self, f1, f2, zmin, zmax):
        # Cut Lyman-alpha forest region
        lyman_alpha_ind = np.logical_and(self.wave >= f1 * (1+self.z_qso), \
            self.wave <= f2 * (1+self.z_qso))
        # Cut analysis boundaries
        forest_boundary = np.logical_and(self.wave >= LYA_WAVELENGTH*(1+zmin), \
            self.wave <= LYA_WAVELENGTH*(1+zmax))
        lyman_alpha_ind = np.logical_and(lyman_alpha_ind, forest_boundary)

        self.wave  = self.wave[lyman_alpha_ind]
        self.flux  = self.flux[lyman_alpha_ind]
        self.error = self.error[lyman_alpha_ind]
        self.mask  = self.mask[lyman_alpha_ind]
        self.size  = len(self.wave)

    def applyMask(self, good_pixels=None, removePixels=True):
        if good_pixels is None:
            good_pixels = self.mask

        if removePixels:
            self.wave  = self.wave[good_pixels]
            self.flux  = self.flux[good_pixels]
            self.error = self.error[good_pixels]
            self.mask  = np.ones_like(self.flux, dtype=np.bool)

            self.size = len(self.wave)
        else:
            self.flux[~good_pixels]  = 0
            self.error[~good_pixels] = 1e10
    
    def setOutliersMask(self, sigma=2.5):
        sigma = np.abs(sigma)
        high_perc = scipy_norm.cdf(sigma)*100
        low_perc  = scipy_norm.cdf(-sigma)*100

        lp_flux, hp_flux = np.percentile(self.flux, [low_perc, high_perc])
        hp_error = np.percentile(self.error, high_perc)

        flux_within_perc  = np.logical_and(self.flux > lp_flux, self.flux < hp_flux)
        error_within_perc = self.error < hp_error

        good_pixels = np.logical_and(flux_within_perc, error_within_perc)

        self.mask = np.logical_and(good_pixels, self.mask)

    def applyMaskDLAs(self, scale=1.0, removePixels=True):
        if self.z_dlas:
            self.mask_dla = np.ones_like(self.wave, dtype=bool)

            for (zd, nhi) in zip(self.z_dlas, self.nhi_dlas):
                lobs = (1+zd) * LYA_WAVELENGTH
                wi = equivalentWidthDLA(10**nhi, zd)*scale
                dla_ind  = np.logical_and(self.wave>lobs-wi/2, self.wave<lobs+wi/2)
                self.mask_dla[dla_ind] = 0

            self.applyMask(self.mask_dla, removePixels)

    def setZScoreMask(self, thres=3.5):
        zsc_mask = np.abs(scipy_zscore(self.flux))<thres
        zsc_mask = np.logical_and(zsc_mask, np.abs(scipy_zscore(self.error))<thres)
        self.mask = np.logical_and(zsc_mask, self.mask)

    def getS2NLya(self, lya_lower=LYA_FIRST_WVL, lya_upper=LYA_LAST_WVL):            
        lyman_alpha_ind = np.logical_and(self.wave >= LYA_FIRST_WVL*(1+self.z_qso), \
            self.wave <= LYA_LAST_WVL*(1+self.z_qso))
        
        temp = 1. / self.error[lyman_alpha_ind & self.mask]

        if len(temp) == 0:
            return -1
        else:
            return np.mean(temp)

    def saveAsBQ(self, fname):
        tbq = BinaryQSO(fname, 'w')
        tbq.save(self.wave, self.flux, self.error, self.size, self.z_qso, \
            self.dec, self.ra, self.s2n, self.specres, self.dv)
        tbq.close()

class BinaryQSO:
    """
    Read and write in binary format for quadratic estimator.

    Parameters
    ----------
    fname : str
        Filename to read or write.
    rw : char
        Mode to open file. Can be either r or w.
    
    __init__(self, fname, rw)
        Opens a file buffer.

    Attributes
    ----------
    file : File
        File buffer. Closed after read or save methods.

    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.
    N : int
        Length of these arrays.
    
    z_qso : float
        Emission redshift of the quasar.
    dv : float
        Pixel width of the spectrum.
    specres : int
        Spectral resolution of the instrument.
    s2n : float
        Signal to noise ratio of the entire spectrum given by KODIAQ.
    ra : float
        ra in radians
    dec : float
        dec in radians

    Methods
    -------
    save(wave, flux, error, N, z_qso, dec, ra, s2n, specres, dv)
        Saves the given parameters in binary format. Does not hold them as attributes.

    read()
        Reads the file. Saves as attributes.
        N : int
            Number of pixels
        z_qso : double
            Emission redshift of the quasar.
        dec : double
        ra : double
            Declination and right ascension in radians
        specres : int
            Dimensionless resolving power.
        s2n : double
            Signal to noise
        dv : double
            Pixel width in km/s
        low_ob_l : double 
            Lower observed wavelength
        upp_ob_l : double
            Upper observed wavelength
        low_re_l : double
            Lower rest frame wavelength
        upp_re_l : double
            Upper rest frame wavelength
        wave : Array of N doubles
        flux : Array of N doubles
        error : Array of N doubles
    """

    def __init__(self, fname, rw):
        self.file  = open(fname, mode=rw + 'b')
    
    def close(self):
        self.file.close()

    def save(self, wave, flux, error, N, z_qso, dec, ra, s2n, specres, dv): 
        # Set up binary data
        low_ob_l = wave[0]
        upp_ob_l = wave[-1]

        low_re_l = low_ob_l / (1. + z_qso)
        upp_re_l = upp_ob_l / (1. + z_qso)

        hdr = struct.pack('idddidddddd', N, z_qso, dec, ra, specres, s2n, dv, \
                          low_ob_l, upp_ob_l, low_re_l, upp_re_l)
        wave_bin = struct.pack('d' * N, *wave)
        flux_bin = struct.pack('d' * N, *flux)
        nois_bin = struct.pack('d' * N, *error)

        # Save to file
        self.file.write(hdr)
        self.file.write(wave_bin)
        self.file.write(flux_bin)
        self.file.write(nois_bin)
        self.file.close()

    def saveas(self, fname):
        new_file = open(fname, mode='wb')

        # Set up binary data
        low_ob_l = self.wave[0]
        upp_ob_l = self.wave[-1]

        low_re_l = low_ob_l / (1. + self.z_qso)
        upp_re_l = upp_ob_l / (1. + self.z_qso)

        hdr = struct.pack('idddidddddd', self.N, self.z_qso, self.dec, self.ra, self.specres, \
            self.s2n, self.dv, low_ob_l, upp_ob_l, low_re_l, upp_re_l)
        wave_bin = struct.pack('d' * self.N, *self.wave)
        flux_bin = struct.pack('d' * self.N, *self.flux)
        nois_bin = struct.pack('d' * self.N, *self.error)

        # Save to file
        new_file.write(hdr)
        new_file.write(wave_bin)
        new_file.write(flux_bin)
        new_file.write(nois_bin)
        new_file.close()

    def read(self):
        header_fmt  = 'idddidddddd'
        header_size = struct.calcsize(header_fmt)

        d = self.file.read(header_size)

        self.N, self.z_qso, self.dec, self.ra, \
        self.specres, self.s2n, self.dv, \
        low_ob_l, upp_ob_l, low_re_l, upp_re_l  = struct.unpack(header_fmt, d)

        array_fmt  = 'd' * self.N
        array_size = struct.calcsize(array_fmt)

        d           = self.file.read(array_size)
        self.wave   = np.array(struct.unpack(array_fmt, d), dtype=np.double)
        d           = self.file.read(array_size)
        self.flux   = np.array(struct.unpack(array_fmt, d), dtype=np.double)
        d           = self.file.read(array_size)
        self.error  = np.array(struct.unpack(array_fmt, d), dtype=np.double)
        
        self.file.close()

class ConfigQMLE:
    """ConfigQMLE reads config.param for the estimator and sets up k & z bins
    
    Attributes
    ----------
    parameters : ConfigParser section
        You can directly access to variables using config.param keys. Note they are str.

    k_0 : float
    k_nlin : int
    k_nlog : int
    k_dlin : float
    k_dlog : float
    k_ledge : float
    k_edges : np.array
    k_bins : np.array

    z_0 : float
    z_n : int
    z_d : float
    z_bins : np.array
    z_edges : np.array
    
    qso_list : str
    qso_dir : str

    sq_vlength : float
    sq_dvgrid : float
    """
    def _getKBins(self):
        self.k_0     = float(self.parameters['K0'])
        self.k_nlin  = int(self.parameters['NumberOfLinearBins'])
        self.k_nlog  = int(self.parameters['NumberOfLog10Bins'])
        self.k_dlin  = float(self.parameters['LinearKBinWidth'])
        self.k_dlog  = float(self.parameters['Log10KBinWidth'])
        try:
            self.k_ledge = float(self.parameters['LastKEdge'])
        except Exception as e:
            self.k_ledge = 0

        self.k_edges, self.k_bins = formBins(self.k_nlin, self.k_nlog, self.k_dlin, \
            self.k_dlog, self.k_0, self.k_ledge)

    def _getZBins(self):
        self.z_0  = float(self.parameters['FirstRedshiftBinCenter'])
        self.z_n  = int(self.parameters['NumberOfRedshiftBins'])
        self.z_d  = float(self.parameters['RedshiftBinWidth'])

        self.z_bins  = self.z_0 + self.z_d * np.arange(self.z_n)
        self.z_edges = self.z_0 + self.z_d * (np.arange(self.z_n+1)-0.5)

    def __init__(self, filename):
        f = open(filename)
        fdata = "[CONFIG]\n" + f.read()
        f.close()

        cparser = ConfigParser(delimiters=' ')
        cparser.read_string(fdata)
        self.parameters = cparser['CONFIG']

        self._getKBins()
        self._getZBins()

        self.qso_list = self.parameters['FileNameList']
        self.qso_dir = self.parameters['FileInputDir']

        self.sq_vlength = float(self.parameters['VelocityLength'])
        self.sq_dvgrid  = self.sq_vlength / (int(self.parameters['NumberVPoints'])-1)

# ------------------------------------------
# --------------- KODIAQ -------------------
# ------------------------------------------

class KODIAQFits(Spectrum):
    """
    Defining parameters and useful methods for a KODIAQ FITS file. 
    By default it keeps the full spectrum and sets up a mask where error > 0. You can additionally 
    update mask to filter spikes using setOutliersMask method.

    Parameters
    ----------
    kodiaq_dir : str
        Directory of KODIAQ data.
    qso_name : str
        Name of the quasar.
    pi_date : str
        Observation.
    spec_prefix : str
        Prefix for the spectrum file.
    z_qso : float
        Emission redshift of the quasar.

    __init__(self, kodiaq_dir, qso_name, pi_date, spec_prefix, z_qso)
        Reads flux and error files. Constructs logarithmicly spaced wavelength array. 
        Mask is error>0 by default.

    Attributes
    ----------
    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.
    mask : 
        Good pixels on full spectrum.
    size : int
        Length of these arrays.
    
    z_qso : float
        Emission redshift of the quasar.
    specres : int
        Spectral resolution of the instrument.
    dv : float
        Pixel width.
    s2n : float
        Signal to noise ratio of the entire spectrum.
    s2n_kodiaq : float
        Signal to noise ratio of the entire spectrum given by KODIAQ.
    s2n_lya : float
        Signal to noise ratio of the Lya forest. 
        Initial value is 0. Run getS2NLya to calculate this value.
        -1 if there is no Lya coverage for a given spectrum.
    ra : float
        RA in radians
    dec : float
        DECL in radians

    qso_name : str
        Name of the quasar.
    pi_date : str
        Observation.
    spec_prefix : str
        Prefix for the spectrum file
    subdir: str
        Subdirectory that the spectrum lives.
    
    Methods
    -------
    _setWavelengthArray(hdr)
        Set the wavelength array in logarithmic spacing.
    applyMask(good_pixels=None)
        Remove masked values from wave, flux and error. 
        Keeps good_pixels and updates the length the arrays.
    
    setOutliersMask(MEAN_FLUX   = 0.7113803432881693, \
                    SIGMA_FLUX  = 0.37433547084407937, \
                    MEAN_ERROR  = 0.09788299539216311, \
                    SIGMA_ERROR = 0.08333137595138172, \
                    SIGMA_CUT   = 5.)
        Mask pixels outside of a given sigma confidence level.
        Mainly use to remove spikes in the flux and error due to 
        continuum normalization near an echelle order edge.
    setHardFluxMask(low_flux=-0.5, high_flux=1.5)
        Less sophisticated cut to constrain flux values between two numbers.
    
    getWaveChunkIndices(rest_frame_edges)
        For a given wevalength edges in A in the rest frame of the QSO, 
        returns the indeces in the array.
    
    getS2NLya(lya_lower = 1050., lya_upper = 1180.)
        Returns <1/e> in the Lya forest. -1 if no coverage.

    """
    def _setWavelengthArray(self, hdr):
        CRPIX1 = hdr["CRPIX1"]
        CDELT1 = hdr["CDELT1"]
        CRVAL1 = hdr["CRVAL1"]

        self.N    = hdr["NAXIS1"]
        self.wave = (np.arange(self.N) + 1.0 - CRPIX1) * CDELT1 + CRVAL1
        self.wave = np.power(10, self.wave)
        
        self.dv = LIGHT_SPEED * CDELT1 * np.log(10)

    def __init__(self, kodiaq_dir, qso_name, pi_date, spec_prefix, z_qso):
        self.qso_name    = qso_name
        self.pi_date     = pi_date
        self.spec_prefix = spec_prefix

        self.subdir = ospath_join(kodiaq_dir, qso_name, pi_date)
        
        flux_fname = ospath_join(self.subdir,"%s_f.fits" % self.spec_prefix)
        erro_fname = ospath_join(self.subdir,"%s_e.fits" % self.spec_prefix)

        with fitsio.FITS(flux_fname) as kf:
            hdr = kf[0].read_header()
            self.flux = np.array(kf[0].read()*1., dtype=np.double)

        with fitsio.FITS(erro_fname) as ke:
            self.error = np.array(ke[0].read()*1., dtype=np.double)

        c = SkyCoord('%s %s'%(hdr["RA"], hdr["DEC"]), unit=(hourangle, deg)) 

        self.s2n_kodiaq = hdr["SIG2NOIS"]

        self._setWavelengthArray(hdr)

        super().__init__(self.wave, self.flux, self.error, z_qso, hdr["SPECRES"], \
            self.dv, c.ra.radian, c.dec.radian)

    # def setOutliersMask(self, MEAN_FLUX = 0.7113803432881693, SIGMA_FLUX = 0.37433547084407937, \
    #     MEAN_ERROR = 0.09788299539216311, SIGMA_ERROR = 0.08333137595138172, SIGMA_CUT = 5.):

    #     HIGHEST_ALLOWED_FLUX  = MEAN_FLUX  + SIGMA_CUT * SIGMA_FLUX
    #     HIGHEST_ALLOWED_ERROR = MEAN_ERROR + SIGMA_CUT * SIGMA_ERROR
    #     LOWEST_ALLOWED_FLUX   = MEAN_FLUX  - SIGMA_CUT * SIGMA_FLUX
        
    #     flux_within_5sigma  = np.logical_and(self.flux > LOWEST_ALLOWED_FLUX, \
    #         self.flux < HIGHEST_ALLOWED_FLUX)
    #     error_within_5sigma = self.error < HIGHEST_ALLOWED_ERROR

    #     good_pixels = np.logical_and(flux_within_5sigma, error_within_5sigma)
        
    #     self.mask = np.logical_and(good_pixels, self.mask)

    # These are 3 sigma percentile given there are about 20m pixels in all quasars
    def setGlobalOutliersMask(self, lower_perc_flux=-0.5014714665, higher_perc_flux=1.4976650673, \
        higher_perc_error=0.4795617122):
        flux_within_perc  = np.logical_and(self.flux > lower_perc_flux, \
            self.flux < higher_perc_flux)
        error_within_perc = self.error < higher_perc_error

        good_pixels = np.logical_and(flux_within_perc, error_within_perc)
        
        self.mask = np.logical_and(good_pixels, self.mask)

    def setHardFluxMask(self, low_flux=-0.5, high_flux=1.5):
        good_pixels = np.logical_and(self.flux > -0.5, self.flux < 1.5)

        self.mask = np.logical_and(good_pixels, self.mask)

    def getWaveChunkIndices(self, rest_frame_edges):
        return np.searchsorted(self.wave/(1.+self.z_qso), rest_frame_edges)

    def print_details(self):
        print(self.qso_name, self.pi_date, self.spec_prefix, "at", self.z_qso)

class KODIAQ_QSO_Iterator:
    """
    Iterates over QSOs in asu.tsv table for KODIAQ. USe as `for qso in KODIAQ_QSO_Iterator`

    Parameters
    ----------
    kodiaq_dir : str
        Directory of KODIAQ data.
    asu_path : str
        Path to asu.tsv table that contains list of quasars. Obtain from 
        http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/154/114
        table3
    clean_pix : bool
        Removes bad pixels (e<0) and outliers when True. Defualt is True.
    
    __init__(self, kodiaq_dir, asu_path, clean_pix=True)
        Reads asu_table from asu_path. Creates an iterable object.

    Attributes
    ----------
    kodiaq_dir : str
        Directory of KODIAQ data.
    asu_table : astropy.io.ascii
        Stores QSO and properties (such as redshift) from asu.tsv table.
    clean_pix : bool
        Removes bad pixels (e<0) and outliers when True.
    iter_asu_table : iterator
        Itarates over asu table.
    qso_number : int
        Counter for the number of quasars.
    qso_name : str
        Name of the QSO.
    z_qso : float
        Emission redshift of the QSO.
    Olam0, Olam1, Rlam0, Rlam1 : int
        First and last wavelengths in observed and rest frame respectively.
    qso_dir : str
        Directory of the QSO.
    readme_table : astropy.io.ascii
        Stores observation and spec_prefix from a selected QSO's README.tbl.

    Methods
    -------
    __iter__() 
        Returns itself.
    __next__()
        Increase qso_number. Jump to next QSO on the table.

    """
    
    def _set_name_dir_table(self, t):
        self.qso_name = t['KODIAQ']
        self.z_qso    = t['zem']

        self.Olam0    = t['Olam0']
        self.Olam1    = t['Olam1']
        self.Rlam0    = t['Rlam0']
        self.Rlam1    = t['Rlam1']

        self.qso_dir  = ospath_join(self.kodiaq_dir, self.qso_name)

        self.readme_table = ascii.read(ospath_join(self.qso_dir, "README.tbl"))

    def __init__(self, kodiaq_dir, asu_path=TABLE_KODIAQ_ASU, clean_pix=True):
        self.kodiaq_dir = kodiaq_dir
        self.clean_pix  = clean_pix
        self.asu_table  = ascii.read(asu_path, data_start=3)
        
        self.iter_asu_table = iter(self.asu_table)
        self.qso_number = 0
        self._set_name_dir_table(self.asu_table[0])

    def __iter__(self):
        return self

    def __next__(self):
        self.qso_number += 1
        self._set_name_dir_table(next(self.iter_asu_table))

        return self

class KODIAQ_OBS_Iterator:
    """
    Iterates over observations in README.tbl for given QSO in KODIAQ. 
    Use as `for obs in KODIAQ_OBS_Iterator`

    Parameters
    ----------
    kqso_iter : KODIAQ_QSO_Iterator
    
    __init__(self, kqso_iter)
        Creates iterable object from kqso_iter's readme_talbe. Reads the first spectrum.

    Attributes
    ----------
    kqso_iter : KODIAQ_QSO_Iterator
    iter_obs : iterator
        Iterator over observations.
    pi_date : str
        Observation.
    spec_prefix : str
        Prefix for spectrum.
    dr : str
        Data release for KODIAQ.
    kodw0, kodw1 : int
        First and last wavelengths in observed frame for spectral chunk.
    spectrum : KODIAQFits
        Stores wave, flux, error and more.

    Methods
    -------
    _set_pidate_specprefix(t)
        Set pi_date, spec_prefix and dr from table element t.
    readSpectrum()
        Reads spectrum for the current pi_date and spec_prefix.
        Removes bad pixels (e<0) and outliers.
    __iter__() 
        Returns itself.
    __next__()
        Jumps to next observation on the table and reads the spectrum.

    """

    def __init__(self, kqso_iter):
        self.kqso_iter = kqso_iter
        self.iter_obs = iter(self.kqso_iter.readme_table)

        self._set_pidate_specprefix(self.kqso_iter.readme_table[0])
        self.readSpectrum()

    def __iter__(self):
        return self
        
    def __next__(self):
        self._set_pidate_specprefix(next(self.iter_obs))
        self.readSpectrum()

        return self

    def _set_pidate_specprefix(self, t):
        self.pi_date     = t['pi_date']
        self.spec_prefix = t['spec_prefix']
        self.dr          = t['kodrelease']
        self.kodw0       = t['kodwblue']
        self.kodw1       = t['kodwred']

    def readSpectrum(self):
        self.spectrum = KODIAQFits(self.kqso_iter.kodiaq_dir, \
            self.kqso_iter.qso_name, self.pi_date, self.spec_prefix, \
            self.kqso_iter.z_qso)
        self.spectrum.setOutliersMask()

        if self.kqso_iter.clean_pix:
            self.spectrum.applyMask()

    def maxLyaObservation(self, w1=LYA_FIRST_WVL, w2=LYA_LAST_WVL):
        max_s2n_lya = -1
        i     = 0
        max_i = 0

        for obs in self:
            current_s2n_lya = obs.spectrum.getS2NLya(w1, w2)
        
            if current_s2n_lya > max_s2n_lya:
                max_s2n_lya = current_s2n_lya
                max_i = i

            i += 1          

        self._set_pidate_specprefix(self.kqso_iter.readme_table[max_i])
        self.readSpectrum()

        return self.spectrum, max_s2n_lya

class KODIAQMasterTable():
    """
    This class generates a master table for the KODIAQ sample.
    You can read the master table simply by astropy.io.ascii

    """
    def __init__(self, kodiaq_dir, fname, rw='r', asu_path='asu.tsv'):
        self.kodiaq_dir = kodiaq_dir
        self.fname = fname
        if rw == 'r':
            self.master_table = ascii.read(fname)
        elif rw == 'w':
            self.generate(asu_path)
            ascii.write(self.master_table, OUTPUT_TABLE_FNAME, format='tab', fast_writer=False)

    def generate(self, asu_path):
        qso_iter = KODIAQ_QSO_Iterator(self.kodiaq_dir, asu_path)

        qso_names     = []
        observations  = []
        spec_prefixes = []

        emission_z = []
        Olam0s     = []
        Olam1s     = []
        Rlam0s     = []
        Rlam1s     = []
        KODw0s     = []
        KODw1s     = []

        lya_s2ns = []

        data_relases = []

        entire_spec_s2ns    = []
        dimless_specres     = []
        pixel_widths        = []

        right_ascensions    = []
        declinations        = []

        # Read or generate if dont exist
        for qso in qso_iter:
            obs_iter = KODIAQ_OBS_Iterator(qso)

            for obs in obs_iter:
                print(qso.qso_number, qso.qso_name, obs.pi_date, obs.spec_prefix)

                qso_names.append(qso.qso_name)
                observations.append(obs.pi_date)
                spec_prefixes.append(obs.spec_prefix)

                Olam0s.append(qso.Olam0)
                Olam1s.append(qso.Olam1)
                Rlam0s.append(qso.Rlam0)
                Rlam1s.append(qso.Rlam1)
                KODw0s.append(obs.kodw0)
                KODw1s.append(obs.kodw1)

                lya_s2ns.append("%.2f"%obs.spectrum.getS2NLya())

                emission_z.append(qso.z_qso)

                data_relases.append(obs.dr)

                entire_spec_s2ns.append(obs.spectrum.s2n)
                dimless_specres.append(obs.spectrum.specres)
                pixel_widths.append("%.2f"%obs.spectrum.dv)

                right_ascensions.append(obs.spectrum.ra)
                declinations.append(obs.spectrum.dec)


        self.master_table = Table([qso_names, observations, spec_prefixes, \
                            emission_z, Olam0s, Olam1s, Rlam0s, Rlam1s, \
                            KODw0s, KODw1s, \
                            lya_s2ns, data_relases, \
                            entire_spec_s2ns, dimless_specres, pixel_widths, \
                            right_ascensions, declinations], \
                            names=['QSO', 'PI_DATE', 'SPEC_PREFIX', \
                            'Z_EM', 'QSO_Olam0', 'QSO_Olam1', 'QSO_Rlam0', 'QSO_Rlam1', \
                            'OBS_Olam0', 'OBS_Olam1', \
                            'LyaS2N', 'DR', 'S2N', 'SPECRES','dv','RA', 'DE']) 

    def find_qso(self, qso_name):
        return np.where( np.array(self.master_table['QSO']) == qso_name )[0]

def getKODIAQLyaMaxS2NObsList(KODIAQdir, asu_path=TABLE_KODIAQ_ASU):
    qso_iter = KODIAQ_QSO_Iterator(KODIAQdir, asu_path, clean_pix=True)
    spec_list = []

    # Start iterating quasars in KODIAQ sample
    # Each quasar has multiple observations
    # Pick the one with highest signal to noise in Ly-alpha region
    for qso in qso_iter:
        obs_iter = KODIAQ_OBS_Iterator(qso)

        # Pick highest S2N obs
        max_obs_spectrum, maxs2n = obs_iter.maxLyaObservation()

        if maxs2n != -1:
            spec_list.append(max_obs_spectrum)

    return spec_list

# ------------------------------------------
# --------------- XQ-100 -------------------
# ------------------------------------------

class XQ100Fits(Spectrum):
    """Reading class for XQ-100 FITS file. 
    By default it keeps the full spectrum and sets up a mask where error > 0. You can additionally 
    update mask to filter spikes using setOutliersMask method.

    Parameters
    ----------
    filename : str

    __init__(self, filename)
        Reads the spectrum. Find z_qso from the table. Mask is error>0 by default.

    Attributes
    ----------
    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.
    mask : 
        Good pixels on full spectrum.
    size : int
        Length of these arrays.
    
    z_qso : float
        Emission redshift of the quasar.
    specres : int
        Spectral resolution of the instrument.
    dv : float
        Pixel width.
    s2n : float
        Signal to noise ratio of the entire spectrum.
    s2n_lya : float
        Signal to noise ratio of the Lya forest. 
        Initial value is 0. Run getS2NLya to calculate this value.
        -1 if there is no Lya coverage for a given spectrum.
    ra : float
        RA in radians
    dec : float
        DECL in radians

    object : str
        Name of the quasar.
    arm : str
        Spectrograph arm.
    
    Methods
    -------
    applyMask(good_pixels=None)
        Remove masked values from wave, flux and error. 
        Keeps good_pixels and updates the length the arrays.
    
    setOutliersMask(mean_flux=0.6556496616, std_flux=0.4257079242, mean_error=0.0474591657, \
    std_error=0.0732692789, nsigma_cut=5.)
        Mask pixels outside of a given sigma confidence level.Mainly use to remove spikes in the \
        flux and error due to continuum normalization near an echelle order edge.
    setHardCutMask(self, r=-100, fc=-1e-15)
        Cut from Irsic et al 2016. Keeps F>r and f>fc.
    
    """
    specres_interp_uvb = interp1d([0.5, 0.8, 1.0], [9700, 6700, 5400], \
        bounds_error=False, fill_value=(9700, 5400))
    specres_interp_vis = interp1d([0.4, 0.7, 0.9], [18400, 11400, 8900], \
        bounds_error=False, fill_value=(18400, 8900))
    xq100_list_fits = fitsio.FITS(TABLE_XQ100_SUM)[1]
    xq100_dla_csv = ascii.read(TABLE_XQ100_DLA, fill_values="")

    def __init__(self, filename, correctSeeing=True):
        with fitsio.FITS(filename) as xqf:
            hdr0 = xqf[0].read_header()
            data = xqf[1].read()[0]

        self.object = hdr0['OBJECT']
        self.arm    = hdr0['DISPELEM']

        i = XQ100Fits.xq100_list_fits.where("OBJECT == '%s'"%self.object)[0]
        d = XQ100Fits.xq100_list_fits[i]
        z_qso = d['Z_QSO']
        seeing_ave = (d['SEEING_MIN']+d['SEEING_MAX'])/2
        seeing_ave = 1.0 if np.isnan(seeing_ave) else seeing_ave

        c = SkyCoord('%s %s'%(hdr0["RA"], hdr0["DEC"]), unit=deg) 

        wave = data['WAVE'] * 10.
        flux = data['FLUX']
        self.cont = data['CONTINUUM']
        err_flux = data['ERR_FLUX']

        if self.arm == 'VIS':
            dv = 11. # km/s
            specres = int(np.around(XQ100Fits.specres_interp_vis(seeing_ave), decimals=-2)) \
                if correctSeeing else int(hdr0["SPEC_RES"])
        elif self.arm == 'UVB':
            dv = 20. # km/s
            specres = int(np.around(XQ100Fits.specres_interp_uvb(seeing_ave), decimals=-2)) \
                if correctSeeing else int(hdr0["SPEC_RES"])

        super(XQ100Fits, self).__init__(wave, flux/self.cont, err_flux/self.cont, \
            z_qso, specres, dv, c.ra.radian, c.dec.radian)
        
        # d = XQ100Fits.xq100_dla_csv[XQ100Fits.xq100_dla_csv["QSO"] == self.object]
        # d = np.array(d)[0]

        # if d['zabs']!='nan':
        #     self.z_dlas  = [float(z) for z in str(d['zabs']).split(',')]
        #     self.nhi_dlas= [float(n) for n in str(d['logN']).split(',')]

    def setHardCutMask(self, r=-100, fc=-1e-15):
        good_pixels = np.logical_and(self.flux > r, self.flux*self.cont > fc)
        self.mask = np.logical_and(good_pixels, self.mask)
    
    # These are 3 sigma percentile given there are only 2.5m pixels in all quasars
    def setGlobalOutliersMask(self, lower_perc_flux=-0.5062193526, higher_perc_flux=1.4352282962, \
        higher_perc_error=0.6368227610):
        flux_within_perc  = np.logical_and(self.flux > lower_perc_flux, \
            self.flux < higher_perc_flux)
        error_within_perc = self.error < higher_perc_error

        good_pixels = np.logical_and(flux_within_perc, error_within_perc)
        
        self.mask = np.logical_and(good_pixels, self.mask)

# ------------------------------------------
# --------------- UVES ---------------------
# ------------------------------------------

class SQUADFits(Spectrum):
    """Reading class for SQUAD FITS file. 
    By default it keeps the full spectrum and sets up a mask where error > 0. You can additionally 
    update mask to filter spikes using setOutliersMask method.

    Parameters
    ----------
    filename : str

    __init__(self, filename)
        Reads the spectrum. Find z_qso from the table. Mask is error>0 by default.

    Attributes
    ----------
    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.
    mask : 
        Good pixels on full spectrum.
    size : int
        Length of these arrays.
    flag: str
        Should be "0" for good spectrum.
    
    z_qso : float
        Emission redshift of the quasar.
    specres : int
        Spectral resolution of the instrument.
    dv : float
        Pixel width.
    s2n : float
        Signal to noise ratio of the entire spectrum.
    s2n_lya : float
        Signal to noise ratio of the Lya forest. 
        Initial value is 0. Run getS2NLya to calculate this value.
        -1 if there is no Lya coverage for a given spectrum.
    ra : float
        RA in radians
    dec : float
        DECL in radians

    object : str
        Name of the quasar.
    
    Methods
    -------
    applyMask(good_pixels=None)
        Remove masked values from wave, flux and error. 
        Keeps good_pixels and updates the length the arrays.
    
    """

    uves_squad_csv = ascii.read(TABLE_SQUAD_DR1, fill_values="")

    def _seeingCorrection(d, correctSeeing):
        slit_width = np.mean(np.array(d['SlitWidths'].split(","), dtype=np.double))
        seeing_med = slit_width
        print("Slit width: ", slit_width)

        if d['Seeing']:
            tmp = d['Seeing'].split(",")[1]
            if tmp!='NA':
                seeing_med = float(tmp)
        print("Median seeing: ", seeing_med)
        
        if correctSeeing and seeing_med < slit_width:
            return slit_width/seeing_med
        else:
            return 1

    def _lowFluxErrorCorrection(data, corrError, filter_size=5):
        if corrError:
            chi_sq_clip = data['CHACLIP']/(data['NPACLIP']-1+1e-8)
            chi_sq_clip = scipy_median_filter(chi_sq_clip, size=filter_size, mode='reflect')
            chi_sq_clip[chi_sq_clip<1] = 1
            return np.sqrt(chi_sq_clip)
        else:
            return 1

    def __init__(self, filename, correctSeeing=True, corrError=True):
        with fitsio.FITS(filename) as usf:
            hdr0 = usf[0].read_header()
            data = usf[1].read()[0]

        self.object = hdr0['OBJECT']

        d = SQUADFits.uves_squad_csv[SQUADFits.uves_squad_csv["Name_Adopt"] == self.object]
        d = np.array(d)[0]
        z_qso = float(d["zem_Adopt"])
        self.flag = str(d["Spec_status"])
        
        specres = hdr0['SPEC_RES'] * SQUADFits._seeingCorrection(d, correctSeeing)
        specres = int(np.around(specres, decimals=-2))

        c = SkyCoord('%s %s'%(hdr0["RA"], hdr0["DEC"]), unit=deg) 

        wave = data['WAVE']
        flux = data['FLUX']
        self.cont = data['CONTINUUM']
        err_flux = data['ERR'] * SQUADFits._lowFluxErrorCorrection(data, corrError)
        dv = d['Dispersion']
        # dv = np.around(np.median(LIGHT_SPEED*np.diff(np.log(wave))), decimals=1)
                   
        super(SQUADFits, self).__init__(wave, flux, err_flux, \
            z_qso, specres, dv, c.ra.radian, c.dec.radian)

        if d['DLAzabs']:
            self.z_dlas  = [float(z) for z in str(d['DLAzabs']).split(',')]
            self.nhi_dlas= [float(n) for n in str(d['DLAlogNHI']).split(',')]

    def setHardCutMask(self):
        pass

    # These are 3 sigma percentile given there are about 54m pixels in all quasars
    def setGlobalOutliersMask(self, lower_perc_flux=-0.9336882812, higher_perc_flux=2.4265680231, \
        higher_perc_error=2.0480231349):
        flux_within_perc  = np.logical_and(self.flux > lower_perc_flux, \
            self.flux < higher_perc_flux)
        error_within_perc = self.error < higher_perc_error

        good_pixels = np.logical_and(flux_within_perc, error_within_perc)
        
        self.mask = np.logical_and(good_pixels, self.mask)



























