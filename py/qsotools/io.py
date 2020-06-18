import numpy as np
import struct
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
    SPECRES : int
        Spectral resolution of the instrument.
    S2N : float
        Signal to noise ratio of the entire spectrum given by KODIAQ.
    RA : float
        RA in radians
    DECL : float
        DECL in radians

    Methods
    -------
    save(wave, flux, error, N, z_qso, DECL, RA, S2N, SPECRES, dv)
        Saves the given parameters in binary format. Does not hold them as attributes.

    read()
        Reads the file. Saves as attributes and returns.
        N : int
            Number of pixels
        z_qso : double
            Emission redshift of the quasar.
        DECL : double
        RA : double
            Declination and right ascension in radians
        SPECRES : int
            Dimensionless resolving power.
        S2N : double
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

    def save(self, wave, flux, error, N, z_qso, DECL, RA, S2N, SPECRES, dv): 
        # Set up binary data
        low_ob_l = wave[0]
        upp_ob_l = wave[-1]

        low_re_l = low_ob_l / (1. + z_qso)
        upp_re_l = upp_ob_l / (1. + z_qso)

        hdr = struct.pack('idddidddddd', N, z_qso, DECL, RA, SPECRES, S2N, dv, \
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

        hdr = struct.pack('idddidddddd', self.N, self.z_qso, self.DECL, self.RA, self.SPECRES, self.S2N, self.dv, \
                          low_ob_l, upp_ob_l, low_re_l, upp_re_l)
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

        self.N, self.z_qso, self.DECL, self.RA, \
        self.SPECRES, self.S2N, self.dv, \
        low_ob_l, upp_ob_l, low_re_l, upp_re_l  = struct.unpack(header_fmt, d)

        array_fmt  = 'd' * self.N
        array_size = struct.calcsize(array_fmt)

        d           = self.file.read(array_size)
        self.wave   = struct.unpack(array_fmt, d)
        d           = self.file.read(array_size)
        self.flux   = struct.unpack(array_fmt, d)
        d           = self.file.read(array_size)
        self.error  = struct.unpack(array_fmt, d)
        
        self.file.close()

        return  self.N, self.z_qso, self.DECL, self.RA, self.SPECRES, self.S2N, self.dv, \
                low_ob_l, upp_ob_l, low_re_l, upp_re_l, \
                self.wave, self.flux, self.error












