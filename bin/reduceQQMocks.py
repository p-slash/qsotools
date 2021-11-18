#!/usr/bin/env python
import argparse
import fitsio
import glob
import time
import logging
from multiprocessing import Pool

from os      import walk as os_walk, makedirs as os_makedirs
from os.path import join as ospath_join, basename as ospath_base

import numpy as np
from scipy.interpolate import interp1d

import qsotools.fiducial as fid
import qsotools.specops as so
from qsotools.io import saveListByLine
from qsotools.mocklib import lognMeanFluxGH as TRUE_MEAN_FLUX

ARMS = ['B', 'R', 'Z']

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

def saveDelta(thid, wave, delta, ivar, z_qso, ra, dec, rmat, fdelta, args):
    ndiags = rmat.shape[0]

    data = np.zeros(wave.size, dtype=[('LOGLAM','f8'),('DELTA','f8'),('IVAR','f8'), \
        ('RESOMAT','f8', ndiags)])

    data['LOGLAM'] = np.log10(wave)
    data['DELTA']  = delta
    data['IVAR']   = ivar
    data['RESOMAT']= rmat.T
    R_kms = so.fitGaussian2RMat(thid, wave, rmat)

    hdr_dict = {'TARGETID': thid, 'RA': ra/180.*np.pi, 'DEC': dec/180.*np.pi, 'Z': float(z_qso), \
        'MEANZ': np.mean(wave)/fid.LYA_WAVELENGTH -1, 'MEANRESO': R_kms, \
        'MEANSNR': np.mean(np.sqrt(data['IVAR'])), 'LIN_BIN': True, \
        'DLL':np.median(np.diff(data['LOGLAM'])), 'DLAMBDA':np.median(np.diff(wave)) }

    if args.oversample_rmat>1:
        hdr_dict['OVERSAMP'] = args.oversample_rmat

    if not args.nosave:
        fdelta.write(data, header=hdr_dict)

class Reducer(object):
    def openFITSFiles(self, fname):
        rreplace  = lambda s, new: new.join(s.rsplit("/spectra-", 1))
        self.fitsfiles['Spec']  = fitsio.FITS(fname)
        self.fitsfiles['Truth'] = fitsio.FITS(rreplace(fname, "/truth-"))
        self.fitsfiles['Zbest'] = fitsio.FITS(rreplace(fname, "/zbest-"))

        fdname = rreplace(fname, "/delta-")
        if self.args.output_dir != self.args.Directory:
            fdname = ospath_base(fdname)
            fdname = ospath_join(self.args.output_dir, fdname)

        if not args.nosave:
            self.fitsfiles['Delta'] = fitsio.FITS(fdname, "rw", clobber=True)

        self.badspectra_fname = rreplace(fname, "/bad_spectra-")
        removeSuffix = lambda s, suf: s[:-len(suf)] if s.endswith(suf) else s
        self.badspectra_fname = removeSuffix(self.badspectra_fname, ".gz")
        self.badspectra_fname = removeSuffix(self.badspectra_fname, ".fits")
        self.badspectra_fname += ".txt"
        if self.args.output_dir != self.args.Directory:
            self.badspectra_fname = ospath_base(self.badspectra_fname)
            self.badspectra_fname = ospath_join(self.args.output_dir, self.badspectra_fname)

    def closeFITSFiles(self):
        self.fitsfiles['Spec'].close()
        self.fitsfiles['Truth'].close()
        self.fitsfiles['Zbest'].close()
        if not args.nosave:
            self.fitsfiles['Delta'].close()

    def forEachArm(self, arm, fbrmap):
        ARM_WAVE   = self.fitsfiles['Spec'][f'{arm}_WAVELENGTH'].read()
        nspectra   = self.fitsfiles['Spec'][f'{arm}_FLUX'].read_header()['NAXIS2']
        ARM_FLUXES = self.fitsfiles['Spec'][f'{arm}_FLUX'].read()
        ARM_IVAR   = self.fitsfiles['Spec'][f'{arm}_IVAR'].read()
        ARM_MASK   = np.array(self.fitsfiles['Spec'][f'{arm}_MASK'].read(), dtype=bool)
        ARM_RESOM  = self.fitsfiles['Truth'][f'{arm}_RESOLUTION'].read()

        for i in range(nspectra):
            thid  = fbrmap['TARGETID'][i]
            ra    = fbrmap['TARGET_RA'][i]
            dec   = fbrmap['TARGET_DEC'][i]
            z_qso = getRedshift(i, self.fitsfiles['Zbest'])

            # cut out forest, but do not remove masked pixels individually
            # resolution matrix assumes all pixels to be present
            forest_pixels  = getForestAnalysisRegion(ARM_WAVE, z_qso, self.args)
            remaining_pixels = forest_pixels & ~ARM_MASK[i]

            if np.sum(remaining_pixels)<5:
                # Empty spectrum
                continue

            wave = ARM_WAVE[forest_pixels]
            dlambda = np.mean(np.diff(wave))

            # Skip short chunks
            MAX_NO_PIXELS = int((fid.LYA_LAST_WVL-fid.LYA_FIRST_WVL)*(1+z_qso) / dlambda)
            isShort = lambda x: self.args.skip and (np.sum(x) < MAX_NO_PIXELS * self.args.skip)
            if isShort(remaining_pixels):
                # Short chunk
                continue

            cont_interp = getTrueContinuumInterp(i, self.fitsfiles['Truth'])

            z    = wave/fid.LYA_WAVELENGTH-1
            cont = cont_interp(wave)

            flux = ARM_FLUXES[i][forest_pixels] / cont
            ivar = ARM_IVAR[i][forest_pixels] * cont**2

            # Make it delta
            tr_mf = TRUE_MEAN_FLUX(z)
            delta = flux/tr_mf-1
            ivar  = ivar*tr_mf**2

            # Mask by setting things to 0
            mask = ARM_MASK[i][forest_pixels]
            delta[mask] = 0
            ivar[mask]  = 0

            # Cut rmat forest region, but keep individual bad pixel values in
            rmat = np.delete(ARM_RESOM, ~forest_pixels, axis=1)
            if self.args.oversample_rmat>1:
                try:
                    rmat = so.getOversampledRMat(wave.size, rmat, self.args.oversample_rmat)
                except Exception as e:
                    logging.error("Oversampling failed. %s, TARGETID: %d, Npix: %d.", \
                        e.what(), thid, wave.size)
                    self.bad_spectra.append(f"{self.fname}[{arm}_FLUX][{i}]")
                    continue

            # Save it
            if not args.nosave:
                saveDelta(thid, wave, delta, ivar, z_qso, ra, dec, rmat, \
                    self.fitsfiles['Delta'], self.args)

    def __init__(self, args):
        self.args = args
        self.fitsfiles = {}
        self.bad_spectra = []

    def __call__(self, fname):
        self.openFITSFiles(fname)
        self.fname = fname
        fbrmap = self.fitsfiles['Spec']['FIBERMAP']['TARGETID', 'TARGET_RA', 'TARGET_DEC'].read()
        # Reads ARM_FLUX extensions, it helps serialize i/o
        for arm in ARMS:
            self.forEachArm(arm, fbrmap)

        self.closeFITSFiles()

        if self.bad_spectra:
            logging.info("Saving a list of bad spectra to %s.\n", self.badspectra_fname)
            saveListByLine(self.bad_spectra, self.badspectra_fname)

        return 1

class Progress(object):
    """docstring for Progress"""
    def __init__(self, total, percThres=5):
        self.i = 0
        self.total = total
        self.percThres = percThres
        self.last_progress = 0
        self.start_time = time.time()

    def increase(self):
        self.i+=1
        curr_progress = int(100*self.i/self.total)
        print_condition = (curr_progress-self.last_progress >= self.percThres) or (self.i == 0)

        if print_condition:
            etime = (time.time()-self.start_time)/60 # min
            logging.info(f"Progress: {curr_progress}%. Elapsed time {etime:.1f} mins.")
            self.last_progress = curr_progress

def transversePFolder(P, args):
    working_dir   = ospath_join(args.Directory, str(P))
    fname_spectra = glob.glob(ospath_join(working_dir, "*", "spectra-*.fits*"))

    logging.info("Working in directory %s/. There are %d files.", \
        working_dir, len(fname_spectra))
    pcounter = Progress(len(fname_spectra))

    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(Reducer(args), fname_spectra)

        for i in imap_it:
            pcounter.increase()


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
        "Pass >1 to get finely spaced response function.", type=int, default=1)

    parser.add_argument("--skip", help="Skip short chunks lower than given ratio", type=float)
    parser.add_argument("--nproc", type=int, default=None)

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
    else:
        os_makedirs(args.output_dir, exist_ok=True)

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

    Pcounter = Progress(len(args.P_folders), 1)
    for P in args.P_folders:
        transversePFolder(P, args)
        logging.info("===============================================================")
        logging.info("===============================================================")
        logging.info("===============================================================")
        logging.info("One P folder finished.")
        Pcounter.increase()

    logging.info("Done!")















