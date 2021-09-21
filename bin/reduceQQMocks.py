#!/usr/bin/env python
import argparse
import fitsio
import glob
import logging

from os      import walk as os_walk
from os.path import join as ospath_join

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import qsotools.fiducial as fid
from qsotools.mocklib import lognMeanFluxGH as TRUE_MEAN_FLUX

ARMS = ['B', 'R', 'Z']

def transversePFolder(P, args):
    working_dir   = ospath_join(args.Directory, str(P))
    fname_spectra = glob.iglob(ospath_join(working_dir, "*", "spectra-*.fits*"))

    for fname in fname_spectra:
        fspec  = fitsio.FITS(fname)
        ftruth = fitsio.FITS(fname.replace("/spectra-", "/truth-"))
        fzbest = fitsio.FITS(fname.replace("/spectra-", "/zbest-"))
        fdelta = fitsio.FITS(fname.replace("/spectra-", "/delta-"), "rw", clobber=True)

        fbrmap = fspec['FIBERMAP']['TARGET_RA', 'TARGET_DEC'].read()

        # Reads ARM_FLUX extensions, it helps serialize i/o
        for arm in ARMS:
            forEachArm(arm, fbrmap, fspec, fzbest, ftruth, fdelta, args)

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
    forst_bnd = np.logical_and(wave >= fid.LYA_WAVELENGTH*(1+args.z_forest_min), \
            wave <= fid.LYA_WAVELENGTH*(1+args.z_forest_max))
    lya_ind = np.logical_and(lya_ind, forst_bnd)

    return lya_ind

def fitGaussian2RMat(wave, rmat):
    v  = fid.LIGHT_SPEED * np.log(wave)
    dv = np.mean(np.diff(v))

    fitt = lambda x, R_kms: fid.getSpectographWindow_x(x, \
        fid.LIGHT_SPEED/R_kms/fid.ONE_SIGMA_2_FWHM, dv)

    rmat_ave = np.mean(rmat, axis=1)
    ndiags = rmat_ave.shape[0]
    x = np.arange(ndiags//2,-(ndiags//2)-1,-1)*dv
    R_kms, _ = curve_fit(fitt, x, rmat_ave, p0=dv, bounds=(dv/100, 100*dv))

    logging.debug("Fitting R (km/s) to the average resomat.")
    logging.debug("ndiags=%d, R_kms=%.1f.", ndiags, R_kms)

    return R_kms

def saveDelta(wave, delta, ivar, z_qso, ra, dec, rmat, fdelta, args):
    ndiags = rmat.shape[0]

    data = np.zeros(wave.size, dtype=[('LOGLAM','f8'),('DELTA','f8'),('IVAR','f8'),
                          ('RESOMAT','f8', ndiags)])

    data['LOGLAM'] = np.log10(wave)
    data['DELTA']  = delta
    data['IVAR']   = ivar
    data['RESOMAT']= rmat
    R_kms = fitGaussian2RMat(wave, rmat)

    hdr_dict = {'RA': ra/180.*np.pi, 'DEC': dec/180.*np.pi, 'Z': float(z_qso), \
        'MEANZ': np.mean(wave)/fid.LYA_WAVELENGTH -1, 'MEANRESO': R_kms, \
        'MEANSNR': np.mean(np.sqrt(data['IVAR'])), 
        'DLL':np.median(np.diff(data['LOGLAM']))}

    fdelta.write(data, header=hdr_dict)

def forEachArm(arm, fbrmap, fspec, fzbest, ftruth, fdelta, args):
    ARM_WAVE   = fspec[f'{arm}_WAVELENGTH'].read()
    nspectra   = fspec[f'{arm}_FLUX'].read_header()['NAXIS2']
    ARM_FLUXES = fspec[f'{arm}_FLUX'].read()
    ARM_IVAR   = fspec[f'{arm}_IVAR'].read()
    ARM_MASK   = fspec[f'{arm}_MASK'].read()
    ARM_RESOM  = ftruth[f'{arm}_RESOLUTION'].read()

    for i in range(nspectra):
        ra    = fbrmap['TARGET_RA']
        dec   = fbrmap['TARGET_DEC']
        z_qso = getRedshift(i, fzbest)

        # cut out forest
        remaining_pixels  = getForestAnalysisRegion(ARM_WAVE, z_qso, args)
        remaining_pixels &= ~ARM_MASK[i]

        # Instead check if short
        if not remaining_pixels.any():
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

        cont_interp = getTrueContinuumInterp(i, ftruth)

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

        # Save it
        saveDelta(wave, delta, ivar, z_qso, ra, dec, rmat, fdelta, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Directory", help="Directory. Saves next to spectra-X-X.fits as delta-X-X.fits")
    parser.add_argument("--P-folders", nargs='*', type=int, help="P folders to run test."
        " Default is all available folders. Leave space between numbers when specifying multiple.")

    # parser.add_argument("--desi-w1", help=("Lower wavelength of DESI wave grid in A. "\
    #     "Default: %(default)s A"), type=float, default=3600.)
    # parser.add_argument("--desi-w2", help=("Higher wavelength of DESI wave grid in A. "\
    #     "Default: %(default)s A"), type=float, default=9800.)
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=1.9)
    parser.add_argument("--z-forest-max", help="Upper end of the forest. Default: %(default)s", \
        type=float, default=4.3)
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

    logging.basicConfig(filename=ospath_join(args.Directory, 'reduction.log'), \
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















