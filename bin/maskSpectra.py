import numpy as np

from os         import makedirs as os_makedirs
from os.path    import join as ospath_join
from shutil     import copyfile as shutil_copyfile, copy as shutil_copy

from qsotools.io import BinaryQSO

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputDir", help="Input directory to read spectra of Filelist")
    parser.add_argument("FileNameList", help="Txt file containing spectra")
    parser.add_argument("OutputDir", help="Directory to save masked spectra")

    parser.add_argument("--keep-pixels", help="When this passed, masked pixel errors are not touched.", \
        action="store_true")
    parser.add_argument("--seed", type=int, default=2342)
    parser.add_argument("--DLA_probability", "-pdla", type=float, default=0.15, \
        help="Probability of a spectrum having a DLA. Default: %(default)s")
    parser.add_argument("--wave_size", help="Wavelength size to mask. Default: %(default)s A", \
        type=float, default=25.)
    parser.add_argument("--mask-value", help="The value of delta after masking. Default: %(default)s", \
        type=float, default=0)
    parser.add_argument("--nosave", help="Does not save out spectra.", action="store_true")
    args = parser.parse_args()

    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)
    
    RNST = np.random.RandomState(args.seed)
    dla_exist = [True, False]
    dla_prob  = [args.DLA_probability, 1-args.DLA_probability]

    file_list = open(args.FileNameList, 'r')
    header = file_list.readline()

    for fl in file_list:
        loc_fname = ospath_join(args.InputDir, fl.rstrip())
        out_fname = ospath_join(args.OutputDir, fl.rstrip())

        # Randomly pick spectrum having DLA
        if not RNST.choice(dla_exist,p=dla_prob):
            shutil_copyfile(loc_fname, out_fname)
            continue

        try:
            spectrum = bq.BinaryQSO(loc_fname, 'r')
            spectrum.read()
        except:
            print("Problem reading ", loc_fname, flush=True)
            continue

        # Uniform randomly pick central wavelength for the DLA
        wc_dla = RNST.uniform(spectrum.wave[0], spectrum.wave[-1])

        # Mask pixels within args.wave_size/2
        mask_indices = np.logical_and(spectrum.wave <= (wc_dla + args.wave_size/2), \
            spectrum.wave >= (wc_dla - args.wave_size/2))
        
        spectrum.flux[mask_indices] = args.mask_value

        if not args.keep_pixels:
            spectrum.error[mask_indices] = 1e10
        
        if not args.nosave:
            try:
                spectrum.saveas(out_fname)
            except:
                print("Problem saving ", out_fname, flush=True)

    # Save the list of files in a txt
    # temp_fname = ospath_join(args.OutputDir, "file_list_qso.txt")
    temp_fname=shutil_copy(args.FileNameList, args.OutputDir)
    print("Saving chunk spectra file list as ", temp_fname)

