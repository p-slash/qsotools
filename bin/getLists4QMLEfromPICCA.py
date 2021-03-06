#!/usr/bin/env python

import numpy as np
import fitsio
import argparse
import glob
from multiprocessing import Pool

from os import makedirs as os_makedirs
from os.path import join as ospath_join, basename as ospath_base

from qsotools.fiducial import LIGHT_SPEED, ONE_SIGMA_2_FWHM
from qsotools.io import saveListByLine
from qsotools.specops import getOversampledRMat

LN10 = np.log(10)

def roundSpecRes(Rkms, dll):
    if dll:
        dv = int(np.round(dll*LIGHT_SPEED*LN10/5)*5)
    else:
        dv = int(np.round(Rkms/5)*5);

    Rint = int(LIGHT_SPEED/Rkms/ONE_SIGMA_2_FWHM/100 + 0.5)*100

    return Rint, dv

class GetNCopy(object):
    def __init__(self, args):
        self.args = args
        if self.args.remove_targetid_list:
            self.ids_to_remove = np.loadtxt(self.args.remove_targetid_list, dtype=int)
        else:
            self.ids_to_remove = None

    def _isIDexcluded(self, hdr):
        if self.ids_to_remove is None:
            return False

        keys = hdr.keys()
        if 'MOCKID' in keys:
            tid = hdr['MOCKID']
        elif 'TARGETID' in keys:
            tid = hdr['TARGETID']
        elif 'THING_ID' in keys:
            tid = hdr['THING_ID']
        else:
            raise Exception("No ID key found in header.")

        return tid in self.ids_to_remove

    def getFlistFromOne(self, f):
        fts = fitsio.FITS(f)

        # if args.oversample_rmat > 1:
        #     f2 = ospath_join(args.osamp_dir, ospath_base(f))
        #     newfits = fitsio.FITS(f2, 'rw', clobber=True)

        i=0
        flst = []
        slst = set()
        for hdu in fts[1:]:
            i+=1
            hdr = hdu.read_header()

            # S/N cut using MEANSNR in header
            if hdr['MEANSNR'] < self.args.snr_cut:
                continue
            # Exclude if in list
            if self._isIDexcluded(hdr):
                continue

            if 'DLL' in hdr.keys():
                dll = hdr['DLL']
            else:
                dll = None

            Rint, dv = roundSpecRes(hdr['MEANRESO'], dll)

            slst.add((Rint, dv))
            flst.append(f"{f}[{i}]")

            # if args.oversample_rmat > 1:
            #     data = hdu.read()

            #     newRmat = getOversampledRMat(data['RESOMAT'].T, args.oversample_rmat)
            #     newdata = np.empty(data.size, dtype=[('LOGLAM','f8'),('DELTA','f8'),('IVAR','f8'),
            #         ('RESOMAT','f8', newRmat.shape[0])])

            #     newdata['LOGLAM']  = data['LOGLAM']
            #     newdata['DELTA']   = data['DELTA']
            #     newdata['IVAR']    = data['IVAR']
            #     newdata['RESOMAT'] = newRmat.T

            #     hdr['OVERSAMP'] = args.oversample_rmat

            #     newfits.write(newdata, header=hdr)

        fts.close()
        # if args.oversample_rmat > 1:
        #     newfits.close()

        return flst, slst

    def __call__(self, f):
        try:
            return self.getFlistFromOne(f)
        except Exception as e:
            print(e)
            return [], set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Directory", help="Directory.")
    parser.add_argument("--out-suffix", default="0")
    parser.add_argument("--flist", nargs='*', help="Only convert these delta files in Directory.")
    parser.add_argument("--snr-cut", help="S/N cut using MEANSNR in header.", default=0, type=float)
    parser.add_argument("--remove-targetid-list", help="txt file with targetid to exclude from final list.")
    # parser.add_argument("--oversample-rmat", type=int, default=1, 
    #     help="Oversampling factor for resolution matrix. "\
    #     "Pass >1 to get finely space response function. It will save to osamp-dir")
    # parser.add_argument("--osamp-dir", help="Folder to save new oversampled resomat.")
    parser.add_argument("--nproc", type=int, default=None)
    args = parser.parse_args()

    # if args.oversample_rmat>1:
    #     if not args.osamp_dir:
    #         args.osamp_dir = ospath_join(args.Directory, "oversampled-deltas")

    #     if args.osamp_dir == args.Directory:
    #         args.osamp_dir = ospath_join(args.Directory, "oversampled-deltas")

    #     os_makedirs(args.osamp_dir, exist_ok=True)

    if args.flist:
        all_deltas = [ospath_join(args.Directory, x) for x in args.flist]
    else:
        all_deltas = glob.iglob(ospath_join(args.Directory, "delta-*.fits*"))
    all_slst = set()
    all_flst = []

    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(GetNCopy(args), all_deltas)

        for (flst, slst) in imap_it:
            all_flst.extend(flst)
            all_slst = all_slst.union(slst)

    temp_fname = ospath_join(args.Directory, f"specres_list-{args.out_suffix}.txt")
    print("Saving spectral resolution values as ", temp_fname)
    saveListByLine(all_slst, temp_fname)

    temp_fname = ospath_join(args.Directory, f"fname_list-{args.out_suffix}.txt")
    print("Saving spectra file list as ", temp_fname)
    saveListByLine(all_flst, temp_fname)
