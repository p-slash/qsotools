import numpy as np
import scipy.sparse
from scipy.stats import binned_statistic, norm
from scipy.signal import fftconvolve
from scipy.linalg import sqrtm as sla_sqrtm
from scipy.special import erf

from astropy.io import ascii
from astropy.table import Table

from qsotools.fiducial import LIGHT_SPEED, LYA_FIRST_WVL, LYA_LAST_WVL, LYA_WAVELENGTH, ONE_SIGMA_2_FWHM

# dv should be in km/s, which is defined as c*dln(lambda)
def createEdgesFromCenters(wave_centers, dv=None, logspacing=True):
    npix = len(wave_centers)
    
    if logspacing:
        if dv is None:
            dv = LIGHT_SPEED * np.min(np.log(wave_centers[1:]/wave_centers[0]) / np.arange(1, npix))
    
        exp_pix = np.exp(dv / LIGHT_SPEED)
    
        wave_edges = 2. / (1. + exp_pix) * wave_centers[0] * np.power(exp_pix, np.arange(npix + 1))
    else:
        dlambda = np.min(wave_centers[1:]-wave_centers[:-1])
        wave_edges = (wave_centers[0] - dlambda/2) + np.arange(npix+1) * dlambda

    return wave_edges

# if new_dv_or_edge is float, then a wavelength grid with given pixel size constructed.
# if new_dv_or_edge is a np.array, then it is used to resample
def resample(wave, flux, error, new_dv_or_edge, keep_empty_bins=False):
    # if new_dv_or_edge is dv
    if isinstance(new_dv_or_edge, float) and new_dv_or_edge > 0:
        dv_c = new_dv_or_edge / LIGHT_SPEED
        new_N = int(np.log(wave[-1] / wave[0]) / dv_c) + 1
    
        new_wave_centers = wave[0] * np.exp(np.arange(new_N) * dv_c)
        new_wave_edges   = createEdgesFromCenters(new_wave_centers, new_dv_or_edge)
    # if new_dv_or_edge is edge array, then logspacing not needed
    elif isinstance(new_dv_or_edge, np.ndarray):
        new_wave_edges   = new_dv_or_edge
        new_wave_centers = (new_wave_edges[1:] + new_wave_edges[:-1]) / 2.
    else:
        raise Exception("Error in resample: new_dv_or_edge should be float or numpy.ndarray")
    
    if error is None: # No error array is given
        binned_flux,  bin_edges, binnumber = binned_statistic(wave, flux, statistic='mean', bins=new_wave_edges)
        empty_bins = np.zeros_like(binned_flux[0], dtype=np.bool)
    elif not error.any(): # error array is all zero
        binned_flux,  bin_edges, binnumber = binned_statistic(wave, flux, statistic='mean', bins=new_wave_edges)
        binned_error = np.zeros_like(binned_flux)
        empty_bins   = np.zeros_like(binned_flux[0], dtype=np.bool)
    else:
        error = np.power(error, -2)
        flux *= error

        binned_flux,  bin_edges, binnumber = binned_statistic(wave, flux, statistic='sum', bins=new_wave_edges)
        binned_error, bin_edges, binnumber = binned_statistic(wave, error, statistic='sum', bins=new_wave_edges)

        binned_flux /= binned_error
        binned_error = 1./np.sqrt(binned_error)

        # Remove empty bins with epsilon error
        empty_bins = np.logical_or(np.abs(binned_flux[0]) < 1e-8, binned_error[0] < 1e-8)
        
    # Remove empty bins from resampled data
    finite_bins = np.logical_and(np.isfinite(binned_flux)[0], ~empty_bins) | keep_empty_bins
    
    new_wave_centers = new_wave_centers[finite_bins]
    binned_flux      = np.array([b[finite_bins] for b in binned_flux])
    
    assert np.size(binned_flux, 1) == len(new_wave_centers)

    if error is not None:
        binned_error = np.array([b[finite_bins] for b in binned_error])
        
        assert np.shape(binned_flux)   == np.shape(binned_error)

        return new_wave_centers, binned_flux, binned_error
    else:
        return new_wave_centers, binned_flux

def divideIntoChunks(wave, flux, error, z_qso, restwave_chunk_edges):
    chunk_indices = np.searchsorted(wave/(1.+z_qso), restwave_chunk_edges)
    
    assert len(np.hsplit(wave, chunk_indices)) == len(chunk_indices)+1

    wave_chunks  = [w for w in np.hsplit(wave, chunk_indices)[1:-1]  if 0 not in w.shape]
    flux_chunks  = [f for f in np.hsplit(flux, chunk_indices)[1:-1]  if 0 not in f.shape]
    error_chunks = [e for e in np.hsplit(error, chunk_indices)[1:-1] if 0 not in e.shape]

    return wave_chunks, flux_chunks, error_chunks

# This function divides a spectrum into
#   3 if len(wave)>no_forest_pixels/2
#   2 if len(wave)>no_forest_pixels/3
# number of pixels in the full forest without gaps
def chunkDynamic(wave, flux, error, no_forest_pixels):
    no_pixels = len(wave)
            
    if no_pixels > 0.66*no_forest_pixels:
        nchunks = 3
    elif no_pixels > 0.33*no_forest_pixels:
        nchunks = 2
    else:
        nchunks = 1
    
    wave  = np.array_split(wave,  nchunks)
    flux  = np.array_split(flux,  nchunks)
    error = np.array_split(error, nchunks)

    return wave, flux, error

# -----------------------------------------------------
# Resolution matrix begins
# -----------------------------------------------------
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

def fitGaussian2RMat(thid, wave, rmat):
    v  = LIGHT_SPEED * np.log(wave)
    dv = np.mean(np.diff(v))

    fitt = lambda x, R_kms: dv*getSpectographWindow_x(x, \
        LIGHT_SPEED/R_kms/ONE_SIGMA_2_FWHM, dv)

    rmat_ave = np.mean(rmat, axis=1)

    # Chi^2 values are bad. Also yields error by std being close 0.
    # rmat_std = np.std(rmat, axis=1)
    # sigma=rmat_std, absolute_sigma=True
    #chi2 = np.sum((fitt(x, R_kms)-rmat_ave)**2/rmat_std**2)

    ndiags = rmat_ave.shape[0]
    x = np.arange(ndiags//2,-(ndiags//2)-1,-1)*dv
    R_kms, eR_kms = curve_fit(fitt, x, rmat_ave, p0=dv, bounds=(dv/10, 10*dv))
    R_kms  = R_kms[0]
    eR_kms = eR_kms[0, 0]

    # Warn if precision or chi^2 is bad
    if eR_kms/R_kms > 0.2:# or chi2/x.size>2:
        logging.debug("Resolution R_kms is questionable. ID: %d", thid)
        logging.debug("R_kms: %.1f km/s - dv: %.1f km/s", R_kms, dv)
        logging.debug("Precision e/R: %.1f percent.", eR_kms/R_kms*100)
        # logging.debug("Chi^2 of the fit: %.1f / %d.", chi2, x.size)

    return R_kms

def constructCSRMatrix(data, oversampling):
    nrows         = data.shape[0]
    nelem_per_row = data.shape[1]
    # assert nelem_per_row % 2 == 1

    ncols = nrows*oversampling + nelem_per_row-1

    indices = np.repeat(np.arange(nrows)*oversampling, nelem_per_row) + \
        np.tile(np.arange(nelem_per_row), nrows)
    iptrs = np.arange(nrows+1)*nelem_per_row

    return scipy.sparse.csr_matrix((data.flatten(), indices, iptrs), shape=(nrows, ncols))

def getDIAfromdata(rmat_data):
    ndiags, nrows = rmat_data.shape
    assert nrows > ndiags

    offsets = np.arange(ndiags//2, -(ndiags//2)-1, -1)
    return scipy.sparse.dia_matrix((rmat_data, offsets), (nrows, nrows))

# Assume offset[0] == -offset[-1]
def getOversampledRMat(wave, rmat, oversampling=3):
    if isinstance(rmat, np.ndarray) and rmat.ndim == 2:
        rmat_dia = getDIAfromdata(rmat)
    elif scipy.sparse.isspmatrix_dia(rmat):
        rmat_dia = rmat
    else:
        raise ValueError("Cannot use given rmat in oversampling.")

    # Properties of the resolution matrix
    nrows = wave.size
    dw    = np.mean(np.diff(wave))
    noff  = rmat_dia.offsets[0]

    # Oversampled resolution matrix elements per row
    nelem_per_row = 2*noff*oversampling + 1
    # ncols = nrows*oversampling + nelem_per_row-1
    
    # Pad the boundaries of the input wave grid
    padded_wave = np.concatenate(( dw*np.arange(-noff, 0)+wave[0], wave, \
        dw*np.arange(1, noff+1)+wave[-1] ))
    # assert padded_wave.size == (2*noff+wave.size)
    
    # Generate oversampled wave grid that is padded at the bndry
    # oversampled_wave = np.linspace(padded_wave[0], padded_wave[-1], \
    #    oversampling*padded_wave.size)
    # assert ncols == oversampled_wave.size

    data = np.zeros((nelem_per_row, nrows))

    # Helper function to pad boundaries
    def getPaddedRow(i):
        row_vector = rmat_dia.getrow(i).data
        if i < noff:
            row_vector = np.concatenate((np.flip(row_vector[i*2+1:]), row_vector))
        if i > nrows-noff-1:
            ii = i-nrows
            row_vector = np.concatenate((row_vector, np.flip(row_vector[:ii*2+1])))
        return row_vector

    for i in range(nrows):
        row_vector = getPaddedRow(i)
        win    = padded_wave[i:i+2*noff+1]
        wout   = np.linspace(win[0], win[-1], nelem_per_row)
        spline = scipy.interpolate.CubicSpline(win, row_vector)

        new_row = spline(wout)
        data[:, i] = new_row/new_row.sum()

    # csr_res = constructCSRMatrix(data, oversampling)
        
    # return csr_res, oversampled_wave
    return data

# -----------------------------------------------------
# Resolution matrix ends
# -----------------------------------------------------

# -----------------------------------------------------
# Mean flux begins
# -----------------------------------------------------
class MeanFluxHist():
    """Object that wraps up binning for mean flux calculation as well as pixel redshift
    histogram.
    """

    def __init__(self, z1, z2, dz=0.1):
        nz = int(np.round((z2-z1)/dz+1))
        self.nz = nz
        self.hist_redshifts = z1 + dz * np.arange(nz)
        self.hist_redshift_edges = z1 + dz * (np.arange(nz+1)-0.5)

        self.total_flux = np.zeros(nz)
        self.total_error2 = np.zeros(nz)
        self.scatter_error = np.zeros(self.nz)
        self.counts = np.zeros(nz+2)
        self.z_hist = np.zeros(nz)

        self.all_flux_values = np.empty((nz,), dtype=list)
        for i in range(nz): self.all_flux_values[i] = []

    def addSpectrum(self, qso, weight=1, f1=LYA_FIRST_WVL, f2=LYA_LAST_WVL, compute_scatter=False):
        lya_ind = np.logical_and(qso.wave>=f1*(1+qso.z_qso), qso.wave<=f2*(1+qso.z_qso))
        z     = qso.wave[lya_ind] / LYA_WAVELENGTH - 1
        flux  = qso.flux[lya_ind]
        error = qso.error[lya_ind]

        fi, _, binnumber = binned_statistic(z, flux, statistic='sum', bins=self.hist_redshift_edges)
        ci = np.bincount(binnumber, minlength=len(self.hist_redshift_edges)+1)
        
        e2i = binned_statistic(z, error**2, statistic='sum', bins=self.hist_redshift_edges)[0]

        # Pixel statistics: Pixel redshift Histogram
        zi, _= np.histogram(z, bins=self.hist_redshift_edges)

        self.z_hist += zi
        self.total_flux += fi * weight
        self.total_error2 += e2i * weight**2
        self.counts += ci * weight

        if compute_scatter:
            for i in range(self.nz): self.all_flux_values[i].extend(flux[binnumber==i+1])

    def getMeanStatistics(self, compute_scatter=False):
        self.mean_flux = self.total_flux / self.counts[1:-1]
        self.mean_error2 = np.sqrt(self.total_error2) / self.counts[1:-1]

        if compute_scatter:
            for i in range(self.nz): self.all_flux_values[i] = np.asarray(self.all_flux_values[i])
            for i in range(self.nz):
                self.scatter_error[i] = np.std(self.all_flux_values[i], ddof=1) \
                    / np.sqrt(self.all_flux_values[i].size)

    def saveHistograms(self, fname_base):
        data = Table([self.hist_redshifts, self.z_hist, self.mean_flux, \
            self.mean_error2, self.scatter_error], \
            names=['z', 'Ncount', 'F-bar', 'sigma_prop', 'sigma_scatter'])
        ascii.write(data, "%s.csv"%fname_base, format='csv', overwrite=True)














