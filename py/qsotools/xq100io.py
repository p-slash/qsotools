import numpy as np
import fitsio
from astropy.coordinates import SkyCoord
from astropy.units import deg as udeg

from qsotools.fiducial import LIGHT_SPEED
from qsotools.io import Spectrum

from pkg_resources import resource_filename
TABLE_XQ100_SUM = resource_filename('qsotools', 'tables/xq100_thework.fits')

class XQ100Fits(Spectrum):
    """docstring for XQ100Fits"""
    xq100_list_fits = fitsio.FITS(TABLE_XQ100_SUM)[1]

    def __init__(self, filename):
        with fitsio.FITS(filename) as xqf:
            hdr0 = xqf[0].read_header()
            data = xqf[1].read()[0]

        self.object = hdr0['OBJECT']
        
        i = XQ100Fits.xq100_list_fits.where("OBJECT == '%s'"%self.object)[0]
        z_qso = XQ100Fits.xq100_list_fits[i]['Z_QSO']
        
        c = SkyCoord('%s %s'%(hdr0["RA"], hdr0["DEC"]), unit=udeg) 

        wave = data['WAVE'] * 10.
        flux = data['FLUX']
        self.cont = data['CONTINUUM']
        err_flux = data['ERR_FLUX']

        dv = np.mean(np.diff(LIGHT_SPEED*np.log(wave)))

        super(XQ100Fits, self).__init__(wave, flux/self.cont, err_flux/self.cont, \
            z_qso, hdr0["SPEC_RES"], dv, c.ra.radian, c.dec.radian)

    def maskHardCut(self, r=-100, fc=-1e-15):
        good_pixels = np.logical_and(self.flux > r, self.flux*self.cont > fc)
        self.mask = np.logical_and(good_pixels, self.mask)