#!/usr/bin/env python

import numpy as np
from astropy.table import Table
from os.path import join as ospath_join
import argparse
import struct

import qsotools.mocklib as lm
import qsotools.fiducial as fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Outputdir", help="Output directory")
    parser.add_argument(
        "--z1", help="First redshift bin center. Default: %(default)s",
        type=float, default=1.9)
    parser.add_argument(
        "--z2", help="First redshift bin center. Default: %(default)s",
        type=float, default=4.4)
    parser.add_argument(
        "--nz", help="Number of redshift bins. Default: %(default)s",
        type=int, default=10000)
    parser.add_argument(
        "--fg08", help="Generate FG 2008 mean flux instead",
        action="store_true")
    args = parser.parse_args()

    # Parameters
    redshifts = np.linspace(args.z1, args.z2, args.nz)
    number_of_z_bins = args.nz

    what_mocks = "fg08" if args.fg08 else "logn"

    if args.fg08:
        print("Using FG 08 mean flux function")
        mean_flux_function = fid.meanFluxFG08
    else:
        print("Using analytic lognormal flux")
        mean_flux_function = lm.lognMeanFluxGH

    mean_fluxs = mean_flux_function(redshifts)

    fname = "mean_flux_%s_z%.1f-%.1f_%dzbins.dat" \
        % (what_mocks, redshifts[0], redshifts[-1], number_of_z_bins)

    # Save data in table
    fname_mean_flux = ospath_join(args.Outputdir, fname)

    mean_flux_table = Table([redshifts, mean_fluxs], names=('z', 'meanF'))

    mean_flux_table.write(fname_mean_flux, format='ascii.fixed_width',
                          formats={'z': '%.10e', 'meanF': '%.10e'},
                          overwrite=True)

    # Write binary
    file_bn = open(fname_mean_flux[:-3] + "bin", "wb")
    file_bn.write(struct.pack('i', number_of_z_bins))
    redshifts.tofile(file_bn)
    mean_fluxs.tofile(file_bn)
    file_bn.close()

    print("Saved binary .bin and ascii table .dat as ", fname_mean_flux)
