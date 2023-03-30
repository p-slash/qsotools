#!/usr/bin/env python
import argparse
from os import makedirs as os_makedirs
from os.path import join as ospath_join
from shutil import copy as shutil_copy

import numpy as np
from scipy.interpolate import interp1d
from astropy.io import ascii

import qsotools.mocklib as lm
import qsotools.fiducial as fid
from qsotools.io import BinaryQSO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputDir", help="Input directory.")
    parser.add_argument(
        "FileNameList", help="List of relative path to spectra.")
    parser.add_argument("OutputDir", help="Output directory.")

    parser.add_argument(
        "--reverse",
        help="Convert flux fluctuations back to flux. If sigma passed, it does not convert to flux.",
        action="store_true")

    parser.add_argument("--mean-flux-file", "-mff",
                        help="Table for mean flux.")
    parser.add_argument("--mean-flux-fg08", "-fg08",
                        help="Use FG 08 mean flux.", action="store_true")
    parser.add_argument("--analytic-ln-flux", "-aln",
                        help="Use analytic lognormal flux",
                        action="store_true")
    parser.add_argument("--without-z-evolution", "-noz",
                        help="Turn off redshift evolution by computing mean flux at median z.",
                        action="store_true")
    parser.add_argument("--chunk-mean", "-ch",
                        help="Use mean flux of the chunk.", action="store_true")
    parser.add_argument("--const-mean-flux", "-cmf",
                        help="Constant mean flux.", type=float)

    parser.add_argument("--seed", type=int, default=2342)
    parser.add_argument("--sigma-eta0", "-s0",
                        help="Continuum noise amplitude in reverse. Std dev of eta 0. Default: %(default)s",
                        type=float, default=0)
    parser.add_argument("--sigma-eta1", "-s1",
                        help="Continuum noise slope in reverse. Std dev of eta 1. Default: %(default)s A",
                        type=float, default=0)

    args = parser.parse_args()

    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)

    RNST = np.random.RandomState(args.seed)

    # Pick mean flux
    if args.mean_flux_file:
        print("Interpolating file ", args.mean_flux_file, flush=True)
        mflux_table = ascii.read(args.mean_flux_file, format='fixed_width')
        z_array = np.array(mflux_table['z'], dtype=np.double)
        mflux_array = np.array(mflux_table['meanF'], dtype=np.double)

        mean_flux_function = interp1d(z_array, mflux_array)
    elif args.analytic_ln_flux:
        print("Using analytic_ln_flux", flush=True)
        mean_flux_function = lm.lognMeanFluxGH
    elif args.mean_flux_fg08:
        print("Using FG 08 mean flux function", flush=True)
        mean_flux_function = fid.meanFluxFG08
    elif args.chunk_mean and not args.reverse:
        print("Using mean flux from each chunk", flush=True)
    elif args.const_mean_flux:
        print("Mean flux is constant across redshift: ",
              args.const_mean_flux, flush=True)

        def mean_flux_function(z): return args.const_mean_flux
    else:
        print("Pass at least one option! Or reverse conflicts with chunk mean", flush=True)
        exit(0)

    if (args.sigma_eta0 > 0 or args.sigma_eta1 > 0) and args.reverse:
        print("Adding continuum noise with sigma-eta-0=%.2f and sigma-eta-1=%.2f" %
              (args.sigma_eta0, args.sigma_eta1), flush=True)

    file_list = open(args.FileNameList, 'r')
    header = file_list.readline()

    for fl in file_list:
        try:
            spectrum = BinaryQSO(ospath_join(args.InputDir, fl.rstrip()), 'r')
        except:
            print("Problem reading ", ospath_join(
                args.InputDir, fl.rstrip()), flush=True)
            continue

        spectrum_z = np.array(
            spectrum.wave, dtype=np.double) / lm.LYA_WAVELENGTH - 1

        if args.without_z_evolution:
            # Round median redshift to 1 decimal point since this is the true z for uniform mocks
            median_z = spectrum_z[int(spectrum.N / 2)]
            median_z = round(10 * median_z) / 10.

            spectrum_z = np.ones_like(spectrum_z) * median_z

        if args.chunk_mean:
            mean_f = np.mean(spectrum.flux)
            mean_flux_function_array = np.ones_like(spectrum.wave) * mean_f
        else:
            mean_flux_function_array = mean_flux_function(spectrum_z)

        if args.reverse:
            se0 = 0
            se1 = 0

            if args.sigma_eta0 > 0:
                se0 = RNST.normal(0, args.sigma_eta0)
            if args.sigma_eta1 > 0:
                se1 = RNST.normal(0, args.sigma_eta1)

            if args.sigma_eta0 > 0 or args.sigma_eta1 > 0:
                # Add noise to delta directly. This should mimick Slosar et al. 2013 continuum fitting model.
                spectrum.flux = np.array(spectrum.flux, dtype=np.double)
                spectrum.flux += lm.genContinuumError(spectrum.wave, se0, se1)
            else:
                spectrum.flux = (1 + np.array(spectrum.flux,
                                              dtype=np.double)) * mean_flux_function_array
                spectrum.error = np.array(
                    spectrum.error, dtype=np.double) * mean_flux_function_array
        else:
            spectrum.flux = np.array(
                spectrum.flux, dtype=np.double) / mean_flux_function_array - 1
            spectrum.error = np.array(
                spectrum.error, dtype=np.double) / mean_flux_function_array

        try:
            spectrum.saveas(ospath_join(args.OutputDir, fl.rstrip()))
        except:
            print("Problem saving ", ospath_join(
                args.OutputDir, fl.rstrip()), flush=True)

    temp_fname = shutil_copy(args.FileNameList, args.OutputDir)
    print("Saving chunk spectra file list as ", temp_fname)
