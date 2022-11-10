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
from numba import njit

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
        cf_size = int(total_nkz*Nd - (Nd*(Nd-1))/2)

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

@njit("Tuple((f8[:, :], f8[:, :, :]))(f8[:, :], i8, i8, i8, i8)")
def getPSandFisher(v, nk, nd, total_nkz, rem_nz=0):
    nboot = v.shape[0]
    newsize = total_nkz-rem_nz*nk
    power = v[:, :newsize]
    fisher = np.zeros((nboot, newsize, newsize))

    farr = v[:, total_nkz:]

    if nd == 3:
        diag_idx = np.array([0, 1, nk])
    else:
        diag_idx = np.arange(nd)

    fari = 0
    for did in diag_idx:
        for jj in range(newsize-did):
            fisher[:, jj, jj+did] = farr[:, fari+jj]
            fisher[:, jj+did, jj] = farr[:, fari+jj]

        fari += total_nkz-did

    for jj in range(newsize):
        fisher[:, jj, jj] = np.where(fisher[:, jj, jj]==0, 1, fisher[:, jj, jj])

    return power, fisher

@njit("i8[:, :](i8[:, :])")
def getCounts(booted_indices):
    bootnum, no_spectra = booted_indices.shape
   
    counts = np.empty_like(booted_indices)
    for b in range(bootnum):
        counts[b] = np.bincount(booted_indices[b], minlength=no_spectra)

    return counts

def getOneSliceBoot(spectra, booted_indices, nspec,
    Nk, Nd, total_nkz, elems_count,
    remove_last_nz_bins, nboot_per_it):
    boot_counts = getCounts(booted_indices)
    logging.info("    > Generated boot indices.")

    # Allocate memory for matrices
    # total_data = np.empty((nboot_per_it, elems_count))
    total_data = boot_counts @ spectra

    logging.info("    > Calculating bootstrapped inverse Fisher and power...")
    total_power_b4, F = getPSandFisher(total_data, Nk, Nd, total_nkz, remove_last_nz_bins)
    total_power = 0.5 * np.linalg.solve(F, total_power_b4)

    return total_power

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("Bootfile", help="File as described in QMLE.")
    parser.add_argument("--bootnum", default=10000, type=int,
        help="Number of bootstrap resamples.")
    parser.add_argument("--nboot-per-it", default=10000, type=int,
        help="Number of bootstraps to generate per iteration.")
    parser.add_argument("--remove-last-nz-bins", default=0, type=int,
        help="Remove last nz bins to obtain invertable Fisher.")
    parser.add_argument("--seed", help="Seed", default=3422, type=int)
    # parser.add_argument("--save-cov", action="store_true")
    parser.add_argument("--save-powers", action="store_true")
    parser.add_argument("--calculate-originals", action="store_true")
    parser.add_argument("--fbase", default="")
    args = parser.parse_args()

    outdir = ospath_dir(args.Bootfile)
    if outdir == "":
        outdir = "."
    # Set up log
    logging.basicConfig(filename=f"{outdir}/bootstrapping.log",
        level=logging.DEBUG)
    logging.info(" ".join(sys.argv))

    Nk, Nz, Nd, total_nkz, elems_count, nspec = getNumbersfromBootfile(args.Bootfile)
    logging.info("There are %d subsamples.", nspec)
    logging.info("Reading bootstrap dat file.")
    spectra = readBootFile(args.Bootfile, elems_count)

    # Calculate original
    if args.calculate_originals:
        logging.info("Calculating original power.")
        total_data = np.ones((1, nspec)) @ spectra
        total_power_b4, F = getPSandFisher(total_data, Nk, Nd, total_nkz, args.remove_last_nz_bins)
        orig_power = 0.5 * np.linalg.solve(F, total_power_b4)
        # Save power to a file
        # Set up output file
        output_fname = ospath_join(outdir, f"{args.fbase}bootstrap-original-power.txt")
        np.savetxt(output_fname, orig_power)
        logging.info(f"Original power saved as {output_fname}.")
        output_fname = ospath_join(outdir, f"{args.fbase}bootstrap-original-fisher.txt")
        np.savetxt(output_fname, F[0])
        logging.info(f"Original fisher saved as {output_fname}.")

    if args.bootnum <= 0:
        logging.info(f"No bootstraps were asked.")
        exit()

    # Generate bootstrap realizations through indexes
    RND = np.random.default_rng(args.seed)
    newpowersize = total_nkz-args.remove_last_nz_bins*Nk
    total_power = np.empty((args.bootnum, newpowersize))
    n_iter = int(args.bootnum/args.nboot_per_it)

    for jj in range(n_iter):
        logging.info(f"Iteration {jj+1}/{n_iter}.")
        i1 = jj*args.nboot_per_it
        i2 = i1+args.nboot_per_it
        booted_indices = RND.integers(low=0, high=nspec, size=(args.nboot_per_it, nspec))
        total_power[i1:i2] = getOneSliceBoot(spectra, booted_indices, nspec,
            Nk, Nd, total_nkz, elems_count,
            args.remove_last_nz_bins, args.nboot_per_it)

    bootstrap_cov = np.cov(total_power, rowvar=False)
    output_fname = ospath_join(outdir, 
        f"{args.fbase}bootstrap-cov-n{args.bootnum}-s{args.seed}.txt")
    np.savetxt(output_fname, bootstrap_cov)
    logging.info(f"Covariance saved as {output_fname}.")

    if args.save_powers:
        output_fname = ospath_join(outdir, 
            f"{args.fbase}bootstrap-power-n{args.bootnum}-s{args.seed}.txt")
        np.savetxt(output_fname, total_power)
        logging.info(f"Power saved as {output_fname}.")

    

