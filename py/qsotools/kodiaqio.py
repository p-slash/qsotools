import numpy as np
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy.units import hourangle, deg

from os.path import join as ospath_join

from qsotools.fiducial import LIGHT_SPEED
from qsotools.io import Spectrum

class KODIAQFits(Spectrum):
    """
    Defining parameters and useful methods for a KODIAQ FITS file. 
    By default it keeps the full spectrum and sets up a mask where error > 0. You can additionally update mask to
    filter spikes using maskOutliers method.

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
        Reads flux and error files. Constructs logarithmicly spaced wavelength array. Mask is error>0 by default.

    Attributes
    ----------
    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.
    mask : 
        Mask on full spectrum.
    N : int
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
    setWavelengthArray(hdr)
        Set the wavelength array in logarithmic spacing.
    applyMask(good_pixels=None)
        Remove masked values from wave, flux and error. Keeps good_pixels and updates the length the arrays.
    
    maskOutliers(MEAN_FLUX   = 0.7113803432881693, \
                    SIGMA_FLUX  = 0.37433547084407937, \
                    MEAN_ERROR  = 0.09788299539216311, \
                    SIGMA_ERROR = 0.08333137595138172, \
                    SIGMA_CUT   = 5.)
        Mask pixels outside of a given sigma confidence level.
        Mainly use to remove spikes in the flux and error due to 
        continuum normalization near an echelle order edge.
    maskHardFlux(low_flux=-0.5, high_flux=1.5)
        Less sophisticated cut to constrain flux values between two numbers.
    
    getWaveChunkIndices(rest_frame_edges)
        For a given wevalength edges in A in the rest frame of the QSO, 
        returns the indeces in the array.
    
    getS2NLya(lya_lower = 1050., lya_upper = 1180.)
        Returns <1/e> in the Lya forest. -1 if no coverage.

    """
    def setWavelengthArray(self, hdr):
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

        try:
            hdul = fits.open(flux_fname)
        except:
            raise

        hdr = hdul[0].header
        c = SkyCoord('%s %s'%(hdr["RA"], hdr["DEC"]), unit=(hourangle, deg)) 

        self.s2n_kodiaq = hdr["SIG2NOIS"]

        self.setWavelengthArray(hdr)

        self.flux = np.array(hdul[0].data*1., dtype=np.double)
        hdul.close()

        try:
            hdul = fits.open(erro_fname)
        except:
            raise

        self.error = np.array(hdul[0].data*1., dtype=np.double)
        hdul.close()
        
        super().__init__(self.wave, self.flux, self.error, z_qso, hdr["SPECRES"], self.dv, c.ra.radian, c.dec.radian)

    def maskOutliers(self,   MEAN_FLUX   = 0.7113803432881693, SIGMA_FLUX  = 0.37433547084407937, \
        MEAN_ERROR  = 0.09788299539216311, SIGMA_ERROR = 0.08333137595138172, SIGMA_CUT   = 5.):

        HIGHEST_ALLOWED_FLUX  = MEAN_FLUX  + SIGMA_CUT * SIGMA_FLUX
        HIGHEST_ALLOWED_ERROR = MEAN_ERROR + SIGMA_CUT * SIGMA_ERROR
        LOWEST_ALLOWED_FLUX   = MEAN_FLUX  - SIGMA_CUT * SIGMA_FLUX
        
        flux_within_5sigma  = np.logical_and(self.flux > LOWEST_ALLOWED_FLUX, self.flux < HIGHEST_ALLOWED_FLUX)
        error_within_5sigma = self.error < HIGHEST_ALLOWED_ERROR

        good_pixels = np.logical_and(flux_within_5sigma, error_within_5sigma)
        
        self.mask = np.logical_and(good_pixels, self.mask)

    def maskHardFlux(self, low_flux=-0.5, high_flux=1.5):
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
    
    def set_name_dir_table(self, t):
        self.qso_name = t['KODIAQ']
        self.z_qso    = t['zem']

        self.Olam0    = t['Olam0']
        self.Olam1    = t['Olam1']
        self.Rlam0    = t['Rlam0']
        self.Rlam1    = t['Rlam1']

        self.qso_dir  = ospath_join(self.kodiaq_dir, self.qso_name)

        self.readme_table = ascii.read(ospath_join(self.qso_dir, "README.tbl"))

    def __init__(self, kodiaq_dir, asu_path, clean_pix=True):
        self.kodiaq_dir = kodiaq_dir
        self.clean_pix  = clean_pix
        self.asu_table  = ascii.read(asu_path, data_start=3)
        
        self.iter_asu_table = iter(self.asu_table)
        self.qso_number = 0
        self.set_name_dir_table(self.asu_table[0])

    def __iter__(self):
        return self

    def __next__(self):
        self.qso_number += 1
        self.set_name_dir_table(next(self.iter_asu_table))

        return self

class KODIAQ_OBS_Iterator:
    """
    Iterates over observations in README.tbl for given QSO in KODIAQ. USe as `for obs in KODIAQ_OBS_Iterator`

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
    set_pidate_specprefix(t)
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

        self.set_pidate_specprefix(self.kqso_iter.readme_table[0])
        self.readSpectrum()

    def __iter__(self):
        return self
        
    def __next__(self):
        self.set_pidate_specprefix(next(self.iter_obs))
        self.readSpectrum()

        return self

    def set_pidate_specprefix(self, t):
        self.pi_date     = t['pi_date']
        self.spec_prefix = t['spec_prefix']
        self.dr          = t['kodrelease']
        self.kodw0       = t['kodwblue']
        self.kodw1       = t['kodwred']

    def readSpectrum(self):
        self.spectrum = KODIAQFits(self.kqso_iter.kodiaq_dir, \
            self.kqso_iter.qso_name, self.pi_date, self.spec_prefix, \
            self.kqso_iter.z_qso)
        self.spectrum.maskOutliers()

        if self.kqso_iter.clean_pix:
            self.spectrum.applyMask()

    def maxLyaObservation(self):
        max_s2n_lya = -1
        i     = 0
        max_i = 0

        for obs in self:
            current_s2n_lya = obs.spectrum.getS2NLya()
        
            if current_s2n_lya > max_s2n_lya:
                max_s2n_lya = current_s2n_lya
                max_i = i

            i += 1          

        self.set_pidate_specprefix(self.kqso_iter.readme_table[max_i])
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

def getKODIAQLyaMaxS2NObsList(KODIAQdir, asu_path):
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





















