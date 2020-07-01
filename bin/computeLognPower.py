#!/usr/bin/env python3

import numpy as np
from os.path     import join as ospath_join
import struct
import argparse
from astropy.table import Table

import qsotools.mocklib as lm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Outputdir", help="Output directory")

    parser.add_argument("--z1", help="First redshift bin center. Default: %(default)s", type=float, default=1.6)
    parser.add_argument("--nz", help="Number of redshift bins. Default: %(default)s", type=int, default=70)
    parser.add_argument("--deltaz", help="Redshift bin width. Default: %(default)s", type=float, default=0.05)

    parser.add_argument("--ngrid", help="Number of grid points. Default is 2^18", type=int, default=2**18)
    parser.add_argument("--griddv", help="Pixel size of the grid in km/s. Default: %(default)s", type=float, default=3)

    parser.add_argument("--seed", help="Seed to generate random numbers. Default: %(default)s", type=int, default=12123)
    args = parser.parse_args()

    # Parameters
    redshifts = args.z1 + args.deltaz * np.arange(args.nz)
    number_of_z_bins = args.nz

    k_centers, power_sp_z = lm.lognPowerSpGH(redshifts, args.ngrid, args.griddv);
    number_of_k_bins = k_centers.size

    zarr_repeated        = np.repeat(redshifts, number_of_k_bins)
    bin_centers_repeated = np.tile(k_centers, number_of_z_bins)

    # Save data
    fname = "NC%d_dv%.1f_%dzbins" % (args.ngrid, args.griddv, number_of_z_bins)
    fname_power = ospath_join(args.Outputdir, "logn_power_"+fname)

    aln_power_table = Table([zarr_repeated, bin_centers_repeated, power_sp_z.ravel()], names=('z', 'kc', 'P-ALN',), \
        meta={'comments':[ "Analytic expression computed using following parameters:", \
        "Lognormal Mocks", "%d %d" % (number_of_z_bins, number_of_k_bins)]})
    aln_power_table.write(fname_power+".tbl", format='ascii.fixed_width', formats={'z':'%.4f', 'kc':'%e', 'P-ALN':'%e'},\
        overwrite=True)

    # Write binary
    file_bn_aln = open(fname_power+".bin", "wb")
    file_bn_aln.write(struct.pack('i', number_of_k_bins))
    file_bn_aln.write(struct.pack('i', number_of_z_bins))
    redshifts.tofile(file_bn_aln)
    k_centers.tofile(file_bn_aln)
    power_sp_z.tofile(file_bn_aln)
    file_bn_aln.close()

    print("Saved binary .bin and ascii table .tbl as ", fname_power)








































