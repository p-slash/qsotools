#!/usr/bin/env python
import argparse
import fitsio
import glob
import time
import logging

from os      import walk as os_walk
from os.path import join as ospath_join, basename as ospath_base

import numpy as np
import scipy.sparse
from scipy.optimize    import curve_fit
from scipy.interpolate import interp1d

import qsotools.fiducial as fid
from qsotools.mocklib import lognMeanFluxGH as TRUE_MEAN_FLUX

ARMS = ['B', 'R', 'Z']

def transversePFolder(P, args):
    working_dir   = ospath_join(args.Directory, str(P))
    fname_spectra = glob.glob(ospath_join(working_dir, "*", "spectra-*.fits*"))

    logging.info("Working in directory %s", working_dir)

    pixNFinal = len(fname_spectra)
    printProgress.last_progress=0
    for pi, fname in enumerate(fname_spectra):
        printProgress(pi, pixNFinal)
        fitsfiles = openFITSFiles(fname)

        fbrmap = fitsfiles['Spec']['FIBERMAP']['TARGETID', 'TARGET_RA', 'TARGET_DEC'].read()

        # Reads ARM_FLUX extensions, it helps serialize i/o
        for arm in ARMS:
            forEachArm(arm, fbrmap, fitsfiles, args)

        closeFITSFiles(fitsfiles)

    printProgress(pixNFinal, pixNFinal)

def openFITSFiles(fname):
    rreplace  = lambda s, new: new.join(s.rsplit("/spectra-", 1))
    fitsfiles = {}
    fitsfiles['Spec']  = fitsio.FITS(fname)
    fitsfiles['Truth'] = fitsio.FITS(rreplace(fname, "/truth-"))
    fitsfiles['Zbest'] = fitsio.FITS(rreplace(fname, "/zbest-"))

    fdname = rreplace(fname, "/delta-")
    if args.output_dir != args.Directory:
        fdname = ospath_base(fdname)
        fdname = ospath_join(args.output_dir, fdname)

    fitsfiles['Delta'] = fitsio.FITS(fdname, "rw", clobber=True)

    return fitsfiles

def closeFITSFiles(fitsfiles):
    fitsfiles['Spec'].close()
    fitsfiles['Truth'].close()
    fitsfiles['Zbest'].close()
    fitsfiles['Delta'].close()

def printProgress(i, ifinal, percThres=5):
    curr_progress = int(100*i/ifinal)
    print_condition = (curr_progress-printProgress.last_progress >= percThres) or (i == 0)

    if print_condition:
        etime = (time.time()-start_time)/60 # min
        logging.info(f"Progress: {curr_progress}%. Elapsed time {etime:.1f} mins.")
        printProgress.last_progress = curr_progress

    return print_condition

# Simply returns redshift.
# Do more if you want to check for errors etc.
def getRedshift(i, fzbest):
    return fzbest['ZBEST']['Z'][i]

def getTrueContinuumInterp(i, ftruth):
    hdr = ftruth['TRUE_CONT'].read_header()
    w1 = hdr['WMIN']
    w2 = hdr['WMAX']
    dw = hdr['DWAVE']
    n = int((w2-w1)/dw)+1
    w = np.linspace(w1, w2, n)
    C = ftruth['TRUE_CONT'][i][1]

    return interp1d(w, C)

def getForestAnalysisRegion(wave, z_qso, args):
    lya_ind = np.logical_and(wave >= fid.LYA_FIRST_WVL * (1+z_qso), \
        wave <= fid.LYA_LAST_WVL * (1+z_qso))

    w1 = max(fid.LYA_WAVELENGTH*(1+args.z_forest_min), args.desi_w1)
    w2 = min(fid.LYA_WAVELENGTH*(1+args.z_forest_max), args.desi_w2)
    forst_bnd = np.logical_and(wave >= w1, wave <= w2)
    lya_ind = np.logical_and(lya_ind, forst_bnd)

    return lya_ind

def fitGaussian2RMat(thid, wave, rmat):
    v  = fid.LIGHT_SPEED * np.log(wave)
    dv = np.mean(np.diff(v))

    fitt = lambda x, R_kms: dv*fid.getSpectographWindow_x(x, \
        fid.LIGHT_SPEED/R_kms/fid.ONE_SIGMA_2_FWHM, dv)

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

def saveDelta(thid, wave, delta, ivar, z_qso, ra, dec, rmat, fdelta, args):
    ndiags = rmat.shape[0]

    data = np.zeros(wave.size, dtype=[('LOGLAM','f8'),('DELTA','f8'),('IVAR','f8'), \
        ('RESOMAT','f8', ndiags)])

    data['LOGLAM'] = np.log10(wave)
    data['DELTA']  = delta
    data['IVAR']   = ivar
    data['RESOMAT']= rmat.T
    R_kms = fitGaussian2RMat(thid, wave, rmat)

    hdr_dict = {'TARGETID': thid, 'RA': ra/180.*np.pi, 'DEC': dec/180.*np.pi, 'Z': float(z_qso), \
        'MEANZ': np.mean(wave)/fid.LYA_WAVELENGTH -1, 'MEANRESO': R_kms, \
        'MEANSNR': np.mean(np.sqrt(data['IVAR'])), 'LIN_BIN': 'T', \
        'DLL':np.median(np.diff(data['LOGLAM'])), 'DLAMBDA':np.median(np.diff(wave)) }

    if args.oversample_rmat>1:
        hdr_dict['OVERSAMP'] = args.oversample_rmat

    if not args.nosave:
        fdelta.write(data, header=hdr_dict)

def forEachArm(arm, fbrmap, fitsfiles, args):
    ARM_WAVE   = fitsfiles['Spec'][f'{arm}_WAVELENGTH'].read()
    nspectra   = fitsfiles['Spec'][f'{arm}_FLUX'].read_header()['NAXIS2']
    ARM_FLUXES = fitsfiles['Spec'][f'{arm}_FLUX'].read()
    ARM_IVAR   = fitsfiles['Spec'][f'{arm}_IVAR'].read()
    ARM_MASK   = np.array(fitsfiles['Spec'][f'{arm}_MASK'].read(), dtype=bool)
    ARM_RESOM  = fitsfiles['Truth'][f'{arm}_RESOLUTION'].read()

    for i in range(nspectra):
        thid  = fbrmap['TARGETID'][i]
        ra    = fbrmap['TARGET_RA'][i]
        dec   = fbrmap['TARGET_DEC'][i]
        z_qso = getRedshift(i, fitsfiles['Zbest'])

        # cut out forest
        remaining_pixels  = getForestAnalysisRegion(ARM_WAVE, z_qso, args)
        remaining_pixels &= ~ARM_MASK[i]

        if np.sum(remaining_pixels)<5:
            # Empty spectrum
            continue

        wave = ARM_WAVE[remaining_pixels]
        dlambda = np.mean(np.diff(wave))

        # Skip short chunks
        MAX_NO_PIXELS = int((fid.LYA_LAST_WVL-fid.LYA_FIRST_WVL)*(1+z_qso) / dlambda)
        isShort = lambda x: args.skip and (np.sum(x) < MAX_NO_PIXELS * args.skip)
        if isShort(remaining_pixels):
            # Short chunk
            continue

        cont_interp = getTrueContinuumInterp(i, fitsfiles['Truth'])

        z    = wave/fid.LYA_WAVELENGTH-1
        cont = cont_interp(wave)
        flux = ARM_FLUXES[i][remaining_pixels] / cont
        ivar = ARM_IVAR[i][remaining_pixels] * cont**2

        # Make it delta
        tr_mf = TRUE_MEAN_FLUX(z)
        delta = flux/tr_mf-1
        ivar  = ivar*tr_mf**2

        # Cut rmat
        rmat = np.delete(ARM_RESOM, ~remaining_pixels, axis=1)
        if args.oversample_rmat>1:
            try:
                rmat = getOversampledRMat(wave, rmat, args.oversample_rmat)
            except:
                logging.error("Oversampling failed. TARGETID: %d, Npix: %d.", thid, wave.size)
                continue

        # Save it
        saveDelta(thid, wave, delta, ivar, z_qso, ra, dec, rmat, fitsfiles['Delta'], args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Directory", help="Directory. Saves next to spectra-X-X.fits as delta-X-X.fits")
    parser.add_argument("--output-dir", help="Save results here if passed.")
    parser.add_argument("--P-folders", nargs='*', type=int, help="P folders to run test."
        " Default is all available folders. Leave space between numbers when specifying multiple.")

    parser.add_argument("--desi-w1", help=("Lower wavelength of DESI wave grid in A. "\
        "Avoid boundary. Default: %(default)s A"), type=float, default=3600.)
    parser.add_argument("--desi-w2", help=("Higher wavelength of DESI wave grid in A. "\
        "Avoid boundary. Default: %(default)s A"), type=float, default=9800.)
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=1.9)
    parser.add_argument("--z-forest-max", help="Upper end of the forest. Default: %(default)s", \
        type=float, default=4.3)

    parser.add_argument("--oversample-rmat", help="Oversampling factor for resolution matrix. "\
        "Pass >1 to get finely space response function.", type=int)

    parser.add_argument("--skip", help="Skip short chunks lower than given ratio", type=float)

    # parser.add_argument("--coadd")
    # parser.add_argument("--chunk-dyn",  action="store_true", \
    #     help="Splits spectrum into three chunks if n>2N/3 or into two chunks if n>N/3.")
    # parser.add_argument("--chunk-fixed",  action="store_true", \
    #     help="Splits spectrum into 3 chunks at fixed rest frame wavelengths")
    parser.add_argument("--debug", help="Set logger to DEBUG level.", action="store_true")
    parser.add_argument("--nosave", help="Does not save mocks to output when passed", \
        action="store_true")

    args = parser.parse_args()

    start_time = time.time()

    if not args.output_dir:
        args.output_dir = args.Directory

    logging.basicConfig(filename=ospath_join(args.output_dir, 'reduction.log'), \
        level=logging.DEBUG if args.debug else logging.INFO)

    # The data is organized under P=0, 1, ... (ipix//100) folders
    # Each P folder has ipix folders.
    # Spectrum is Directory/P/ipix/spectra-16-{ipix}.fits. 16 can be N
    # Truth is Directory/P/ipix/truth-16-{ipix}.fits. This has continuum

    if args.P_folders is None:
        args.P_folders = [int(x) for x in next(os_walk(args.Directory))[1] if x.isdigit()]
        logging.info("Transversing all P folders. There are %d many.", len(args.P_folders))
    else:
        # args.P_folders = [x for x in args.P_folders if x.isdigit()]
        logging.info("Transversing given P folders. There are %d many.", len(args.P_folders))

    for P in args.P_folders:
        transversePFolder(P, args)

    logging.info("Done!")















