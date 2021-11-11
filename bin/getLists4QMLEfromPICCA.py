#!/usr/bin/env python

import numpy as np
import fitsio
import argparse
import glob

from os import makedirs as os_makedirs
from os.path import join as ospath_join, basename as ospath_base

from qsotools.fiducial import LIGHT_SPEED, ONE_SIGMA_2_FWHM
from qsotools.io import saveListByLine
from qsotools.specops import getOversampledRMat

LN10 = np.log(10)

def roundSpecRes(Rkms, dll):
    dv = int(np.round(dll*LIGHT_SPEED*LN10/5)*5);
    Rint = int(LIGHT_SPEED/Rkms/ONE_SIGMA_2_FWHM/100 + 0.5)*100;

    return Rint, dv

def getFlistFromOne(f, args):
    fts = fitsio.FITS(f)

    if args.oversample_rmat > 1:
        f2 = ospath_join(args.osamp_dir, ospath_base(f))
        newfits = fitsio.FITS(f2, 'rw', clobber=True)

    i=0
    flst = []
    slst = set()
    for hdu in fts[1:]:
        i+=1
        hdr = hdu.read_header()

        # S/N cut using MEANSNR in header
        if hdr['MEANSNR'] < args.snr_cut:
            continue

        Rint, dv = roundSpecRes(hdr['MEANRESO'], hdr['DLL'])

        slst.add((Rint, dv))
        flst.append(f"{f}[{i}]")

        if args.oversample_rmat > 1:
            data = hdu.read()

            newRmat = getOversampledRMat(10**data['LOGLAM'], data['RESOMAT'].T, \
                args.oversample_rmat)
            newdata = np.empty(data.size, dtype=[('LOGLAM','f8'),('DELTA','f8'),('IVAR','f8'),
                ('RESOMAT','f8', newRmat.shape[0])])

            newdata['LOGLAM']  = data['LOGLAM']
            newdata['DELTA']   = data['DELTA']
            newdata['IVAR']    = data['IVAR']
            newdata['RESOMAT'] = newRmat.T

            newfits.write(newdata, header=hdr)

    fts.close()
    if args.oversample_rmat > 1:
        newfits.close()

    return flst, slst

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Directory", help="Directory.")
    parser.add_argument("--snr-cut", help="S/N cut using MEANSNR in header. ", default=0, type=float)
    parser.add_argument("--oversample-rmat", type=int, default=1, 
        help="Oversampling factor for resolution matrix. "\
        "Pass >1 to get finely space response function. It will save to osamp-dir")
    parser.add_argument("--osamp-dir", help="Folder to save new oversampled resomat.")
    args = parser.parse_args()

    if args.oversample_rmat>1:
        if not args.osamp_dir:
            args.osamp_dir = ospath_join(args.Directory, "oversampled-deltas")

        if args.osamp_dir == args.Directory:
            args.osamp_dir = ospath_join(args.Directory, "oversampled-deltas")

        os_makedirs(args.osamp_dir, exist_ok=True)

    all_deltas = glob.glob(ospath_join(args.Directory, "delta-*.fits*"))
    all_slst = set()
    all_flst = []

    for f in all_deltas:
        flst, slst = getFlistFromOne(f, args)
        all_flst.extend(flst)
        all_slst = all_slst.union(slst)

    temp_fname = ospath_join(args.Directory, "specres_list.txt")
    print("Saving spectral resolution values as ", temp_fname)
    saveListByLine(all_slst, temp_fname)

    temp_fname = ospath_join(args.Directory, "fname_list.txt")
    print("Saving spectra file list as ", temp_fname)
    saveListByLine(all_flst, temp_fname)
