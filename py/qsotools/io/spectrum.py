import numpy as np
from qsotools.fiducial import LIGHT_SPEED, LYA_FIRST_WVL, LYA_LAST_WVL

class Spectrum():
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
    RA : float
        RA in radians
    DECL : float
        DECL in radians

    __init__(self, wave, flux, error, z_qso, specres, dv, ra, dec)
        Creates this spectrum object. Computes S2N.

    Attributes
    ----------
    mask : 
        Mask on full spectrum, error>0 by default.
    size : int
        Length of arrays.
    s2n : float
        Signal to noise ratio of the entire spectrum as ave(1/error).
    s2n_lya : float
        Signal to noise ratio of the Lya forest. -1 if there is no Lya coverage for a given spectrum.

    Methods
    -------
    applyMask(good_pixels=None)
        Remove masked values from wave, flux and error. Keeps good_pixels and updates the length the arrays.

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

    def applyMask(self, good_pixels=None):
        if good_pixels is None:
            good_pixels = self.mask

        self.wave  = self.wave[good_pixels]
        self.flux  = self.flux[good_pixels]
        self.error = self.error[good_pixels]
        self.mask  = np.ones_like(self.flux, dtype=np.bool)

        self.size = len(self.wave)

    def getS2NLya(self, lya_lower=LYA_FIRST_WVL, lya_upper=LYA_LAST_WVL):            
        lyman_alpha_ind = np.logical_and(self.wave >= LYA_FIRST_WVL*(1+self.z_qso), \
            self.wave <= LYA_LAST_WVL*(1+self.z_qso))
        
        temp = 1. / self.error[lyman_alpha_ind & self.mask]

        if len(temp) == 0:
            return -1
        else:
            return np.mean(temp)
    









