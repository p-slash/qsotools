#!/usr/bin/env python
import argparse
import sys
from os.path import join as ospath_join, getsize as ospath_getsize
from multiprocessing import Pool
from itertools import groupby
import glob
import struct
import time
import logging

import numpy as np
import fitsio

def getNfromBootfile(fname):
    with open(fname, "rb") as f:
        N = int(struct.unpack('i', f.read(struct.calcsize('i')))[0])
    return N

# Returns fishers, powers
# fishers.shape = (nspec, N*N)
# powers.shape = (nspec, N)
def readBootFile(fname):
    filesize = ospath_getsize(fname)

    with open(fname, "rb") as bootfile:
        N = int(struct.unpack('i', bootfile.read(struct.calcsize('i')))[0])

        id_size     = struct.calcsize('i')
        fisher_size = struct.calcsize('d'*N*N)
        power_size  = struct.calcsize('d'*N)
        aspec_size  = id_size+fisher_size+power_size
        nspec = int((filesize-struct.calcsize('i'))/aspec_size)

        powers  = np.empty((nspec, N))
        fishers = np.empty((nspec, N*N))

        for ispec in range(nspec):
            temp_data = bootfile.read(aspec_size)
            fishers[ispec] = struct.unpack('d'*N*N, temp_data[id_size:id_size+fisher_size])
            powers[ispec]  = struct.unpack('d'*N, temp_data[id_size+fisher_size:])

    return fishers, powers

def getCounts(booted_indices, bootnum, no_spectra):
    # Does sorting helps with add.at?
    booted_indices.sort(axis=1)
    counts = np.zeros((bootnum, no_spectra), dtype=np.int)
    for i, btdi in enumerate(booted_indices):
        np.add.at(counts[i], btdi, 1)

    return np.transpose(counts)

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


class Booter(object):
    """docstring for Booter"""
    def __init__(self, N, args):
        self.args = args
        self.Nbin = N

    def __call__(self, pe):
        fname = ospath_join(self.args.BootDirectory, f"bootresults-{pe}.dat")

        # Read fishers and powers
        fishers, powers = readBootFile(fname)
        nspec = powers.shape[0]
        assert (powers.shape[1] == self.Nbin)

        # Create zeros for bootstrap results
        this_fisher   = np.zeros((self.args.bootnum, self.Nbin*self.Nbin))
        this_power_b4 = np.zeros((self.args.bootnum, self.Nbin))

        # Each file has a different seed for multiprocessing
        RND = np.random.default_rng(self.args.seed + pe)
        booted_indices = RND.integers(low=0, high=nspec, size=(self.args.bootnum, nspec))
        counts = getCounts(booted_indices, self.args.bootnum, nspec)

        for ispec in range(nspec):
            ci = counts[ispec]

            total_fisher   += fishers[ispec]*ci[:, None]
            total_power_b4 += powers[ispec]*ci[:, None]

        return total_fisher, total_power_b4
        
if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("BootDirectory", help="Directory where bootresults-N.dat are present.")
    parser.add_argument("OutputFile", help="Output file relative to BootDirectory.")
    parser.add_argument("--bootnum", default=1000, type=int, \
        help="Number of bootstrap resamples. Default: %(default)s")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--seed", default=3422, type=int)
    parser.add_argument("--save-cov", action="store_true")
    args = parser.parse_args()

    # Set up log
    logging.basicConfig(filename=ospath_join(args.BootDirectory, 'bootstrapping.log'), \
        level=logging.INFO)
    logging.info(" ".join(sys.argv))

    # Find the number of bootstrapping files
    bootfiles = glob.glob(ospath_join(args.BootDirectory, "bootresults-*.dat"))
    nbootfiles = len(bootfiles)
    logging.info("There are %d bootresults files.", nbootfiles)

    pcounter = Progress(nbootfiles)
    N = getNfromBootfile(bootfiles[0])
    total_fisher   = np.zeros((args.bootnum, N*N))
    total_power_b4 = np.zeros((args.bootnum, N))
    total_power    = np.zeros((args.bootnum, N))

    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(Booter(N, args), range(nbootfiles))

        for (fisher1, power1) in imap_it:
            total_fisher   += fisher1
            total_power_b4 += power1

            pcounter.increase()

    logging.info("Calculating bootstrapped inverse Fisher and power...")
    for bi in range(args.bootnum):
        total_power[bi] = 0.5 * np.linalg.inv(total_fisher[bi].reshape(N,N)) @ total_power_b4[bi]

    # Save power to a file
    # Set up output file
    output_fname = ospath_join(args.BootDirectory, args.OutputFile)
    np.savetxt(output_fname, total_power)
    logging.info("Power saved as ", output_fname)

    # If time allows, run many bootstraps and save its covariance
    # when save-cov passed
    if args.save_cov:
        bootstrap_cov = np.cov(total_power, rowvar=False)
        cov_filename = oospath_join(args.BootDirectory, f"cov-{args.OutputFile}")
        np.savetxt(cov_filename, bootstrap_cov)
        logging.info("Covariance saved as ", cov_filename)

