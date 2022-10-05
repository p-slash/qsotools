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

        i=0
        high_flst = []
        low_flst = []
        slst = set()
        for hdu in fts[1:]:
            i+=1
            hdr = hdu.read_header()

            # Exclude if in list
            if self._isIDexcluded(hdr):
                continue

            if 'DLL' in hdr.keys():
                dll = hdr['DLL']
            else:
                dll = None

            Rint, dv = roundSpecRes(hdr['MEANRESO'], dll)

            # S/N cut using MEANSNR in header
            if hdr['MEANSNR'] < self.args.snr_cut:
                low_flst.append(f"{f}[{i}]")
            else:
                high_flst.append(f"{f}[{i}]")

            slst.add((Rint, dv))

        fts.close()

        return high_flst, low_flst, slst

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
    parser.add_argument("--snr-cut", help="S/N cut using MEANSNR in header.", default=0, type=float)
    parser.add_argument("--remove-targetid-list", help="txt file with targetid to exclude from final list.")
    parser.add_argument("--nproc", type=int, default=None)
    args = parser.parse_args()

    all_slst = set()
    all_high_flst = []
    all_low_flst = []

    all_deltas = glob.iglob(ospath_join(args.Directory, "delta-*.fits*"))
    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(GetNCopy(args), all_deltas)

        for (hflst, lflst, slst) in imap_it:
            all_high_flst.extend(hflst)
            all_low_flst.extend(lflst)
            all_slst = all_slst.union(slst)

    temp_fname = ospath_join(args.Directory, f"specres_list-{args.out_suffix}.txt")
    print("Saving spectral resolution values as ", temp_fname)
    saveListByLine(all_slst, temp_fname)

    if args.snr_cut <= 0 or not all_low_flst:
        temp_fname = ospath_join(args.Directory, f"fname_list-{args.out_suffix}.txt")
        print("Saving spectra file list as ", temp_fname)
        saveListByLine(all_high_flst, temp_fname)
    else:
        temp_fname = ospath_join(args.Directory, f"fname_list-high-{args.out_suffix}.txt")
        print("Saving high snr spectra file list as ", temp_fname)
        saveListByLine(all_high_flst, temp_fname)

        temp_fname = ospath_join(args.Directory, f"fname_list-low-{args.out_suffix}.txt")
        print("Saving low snr  spectra file list as ", temp_fname)
        saveListByLine(all_low_flst, temp_fname)

