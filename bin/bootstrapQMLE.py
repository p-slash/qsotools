#!/usr/bin/env python

# Multiprocessing does not help speed up computation
# Generating bootstrap realizations by file, constains the fluctuations
# Generate one big boot array, files get their parts

import argparse
import sys
from os.path import join as ospath_join, \
    getsize as ospath_getsize, dirname as ospath_dir

import glob
import struct
import time
import logging

import numpy as np
from numba import jit

from qsotools.utils import Progress

def getNumbersfromBootfile(fname):
    filesize = ospath_getsize(fname)

    with open(fname, "rb") as f:
        Nk = int(struct.unpack('i', f.read(struct.calcsize('i')))[0])
        Nz = int(struct.unpack('i', f.read(struct.calcsize('i')))[0])
        Nd = int(struct.unpack('i', f.read(struct.calcsize('i')))[0])

    total_nkz = Nk*Nz

    if Nd == 3:
        cf_size = 3*total_nkz - Nk - 1
    else:
        cf_size = total_nkz*ndiags - (ndiags*(ndiags-1))/2

    elems_count   = cf_size + total_nkz;
    one_data_size = struct.calcsize('d'*elems_count)

    nspec = int((filesize-3*struct.calcsize('i'))/one_data_size)

    return Nk, Nz, Nd, total_nkz, elems_count, nspec

# Returns data
# First Nkz is power, rest is fisher that needs to be reshaped
def readBootFile(fname, N):
    dt = np.dtype(('f8', N))

    with open(fname, "rb") as bootfile:
        spectra = np.fromfile(bootfile, offset=3*struct.calcsize('i'), dtype=dt)

    return spectra

@jit("Tuple((f8[:, :], f8[:, :, :]))(f8[:, :], i8, i8, i8)", nopython=True)
def getPSandFisher(v, nk, nd, total_nkz):
    nboot = v.shape[0]
    power = v[:, :total_nkz]
    fisher = np.zeros((nboot, total_nkz, total_nkz))

    farr = v[:, total_nkz:]

    if nd == 3:
        diag_idx = np.array([0, 1, nk])
    else:
        diag_idx = np.arange(nd)

    fari = 0
    for did in diag_idx:
        for jj in range(total_nkz-did):
            fisher[:, jj, jj+did] = farr[:, fari+jj]
            fisher[:, jj+did, jj] = farr[:, fari+jj]

        fari += total_nkz-did

    return power, fisher

@jit("i8[:, :](i8[:, :])", nopython=True)
def getCounts(booted_indices):
    bootnum, no_spectra = booted_indices.shape
   
    counts = np.empty_like(booted_indices)
    for b in range(bootnum):
        counts[b] = np.bincount(booted_indices[b], minlength=no_spectra)

    return counts
        
if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("Bootfile", help="File as described in QMLE.")
    parser.add_argument("--bootnum", default=1000, type=int, \
        help="Number of bootstrap resamples. Default: %(default)s")
    parser.add_argument("--seed", default=3422, type=int)
    parser.add_argument("--save-cov", action="store_true")
    args = parser.parse_args()

    outdir = ospath_dir(args.Bootfile)
    # Set up log
    logging.basicConfig(filename=ospath_join(outdir, 'bootstrapping.log'), \
        level=logging.INFO)
    logging.info(" ".join(sys.argv))

    Nk, Nz, Nd, total_nkz, elems_count, nspec = getNumbersfromBootfile(args.Bootfile)
    logging.info("There are %d subsamples.", nspec)

    # Generate bootstrap realizations through indexes
    RND            = np.random.default_rng(args.seed)
    booted_indices = RND.integers(low=0, high=nspec, size=(args.bootnum, nspec))
    # Save original estimate to first array
    boot_counts    = np.empty((args.bootnum+1, nspec))
    boot_counts[0] = 1
    boot_counts[1:]= getCounts(booted_indices)
    logging.info(f"Generated boot indices.")

    # Allocate memory for matrices
    total_data = np.empty((args.bootnum+1, elems_count))
    spectra    = readBootFile(args.Bootfile, elems_count)

    total_data = boot_counts @ spectra

    logging.info("Calculating bootstrapped inverse Fisher and power...")
    total_power_b4, F = getPSandFisher(total_data, Nk, Nd, total_nkz)
    total_power = 0.5 * np.linalg.solve(F, total_power_b4)

    # Save power to a file
    # Set up output file
    output_fname = ospath_join(outdir, "bootstrap-original-power.txt")
    np.savetxt(output_fname, total_power[0])
    logging.info(f"Original power saved as {output_fname}.")

    if args.bootnum == 0:
        logging.info(f"Exiting. Only calculated the original power.")
        exit()

    output_fname = ospath_join(outdir, 
        "bootstrap-power-n%d-s%d.txt" % (args.bootnum, args.seed))
    np.savetxt(output_fname, total_power[1:])
    logging.info(f"Power saved as {output_fname}.")

    # If time allows, run many bootstraps and save its covariance
    # when save-cov passed
    if args.save_cov:
        bootstrap_cov = np.cov(total_power[1:], rowvar=False)
        output_fname = ospath_join(outdir, 
            "bootstrap-cov-n%d-s%d.txt" % (args.bootnum, args.seed))
        np.savetxt(output_fname, bootstrap_cov)
        logging.info(f"Covariance saved as {output_fname}.")

