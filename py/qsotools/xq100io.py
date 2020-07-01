import numpy as np
import fitsio
from astropy.coordinates import SkyCoord
from astropy.units import deg as udeg

from qsotools.fiducial import LIGHT_SPEED
from qsotools.io import Spectrum

from pkg_resources import resource_filename
TABLE_XQ100_SUM = resource_filename('qsotools', 'tables/xq100_thework.fits')

class XQ100Fits(Spectrum):
    """Reading class for XQ-100 FITS file. 
    By default it keeps the full spectrum and sets up a mask where error > 0. You can additionally 
    update mask to filter spikes using maskOutliers method.

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
    
    maskOutliers(mean_flux=0.6556496616, std_flux=0.4257079242, mean_error=0.0474591657, \
    std_error=0.0732692789, nsigma_cut=5.)
        Mask pixels outside of a given sigma confidence level.Mainly use to remove spikes in the \
        flux and error due to continuum normalization near an echelle order edge.
    maskHardCut(self, r=-100, fc=-1e-15)
        Cut from Irsic et al 2016. Keeps F>r and f>fc.
    
    """

    xq100_list_fits = fitsio.FITS(TABLE_XQ100_SUM)[1]

    def __init__(self, filename):
        with fitsio.FITS(filename) as xqf:
            hdr0 = xqf[0].read_header()
            data = xqf[1].read()[0]

        self.object = hdr0['OBJECT']
        self.arm    = hdr0['DISPELEM']

        i = XQ100Fits.xq100_list_fits.where("OBJECT == '%s'"%self.object)[0]
        z_qso = XQ100Fits.xq100_list_fits[i]['Z_QSO']
        
        c = SkyCoord('%s %s'%(hdr0["RA"], hdr0["DEC"]), unit=udeg) 

        wave = data['WAVE'] * 10.
        flux = data['FLUX']
        self.cont = data['CONTINUUM']
        err_flux = data['ERR_FLUX']

        if self.arm == 'VIS':
            dv = 11. # km/s
        elif self.arm == 'UVB':
            dv = 20. # km/s

        super(XQ100Fits, self).__init__(wave, flux/self.cont, err_flux/self.cont, \
            z_qso, int(hdr0["SPEC_RES"]), dv, c.ra.radian, c.dec.radian)

    def maskHardCut(self, r=-100, fc=-1e-15):
        good_pixels = np.logical_and(self.flux > r, self.flux*self.cont > fc)
        self.mask = np.logical_and(good_pixels, self.mask)

    def maskOutliers(self, mean_flux=0.6556496616, std_flux=0.4257079242, \
        mean_error=0.0474591657, std_error=0.0732692789, nsigma_cut=5.):

        highest_allowed_flux  = mean_flux  + (nsigma_cut * std_flux)
        highest_allowed_error = mean_error + (nsigma_cut * std_error)
        lowest_allowed_flux   = mean_flux  - (nsigma_cut * std_flux)
        
        flux_within_5sigma  = np.logical_and(self.flux > lowest_allowed_flux, \
            self.flux < highest_allowed_flux)
        error_within_5sigma = self.error < highest_allowed_error

        good_pixels = np.logical_and(flux_within_5sigma, error_within_5sigma)
        
        self.mask = np.logical_and(good_pixels, self.mask)
