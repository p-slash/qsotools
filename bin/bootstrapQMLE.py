#!/usr/bin/env python

# Multiprocessing does not help speed up computation
# Generating bootstrap realizations by file, constains the fluctuations
# Generate one big boot array, files get their parts

import argparse
import sys
from os.path import join as ospath_join, getsize as ospath_getsize
import glob
import struct
import time
import logging

import numpy as np

def getNumbersfromBootfile(fname):
    filesize = ospath_getsize(fname)

    with open(fname, "rb") as f:
        N = int(struct.unpack('i', f.read(struct.calcsize('i')))[0])

        id_size     = struct.calcsize('i')
        fisher_size = struct.calcsize('d'*N*N)
        power_size  = struct.calcsize('d'*N)
        aspec_size  = id_size+fisher_size+power_size
        nspec = int((filesize-struct.calcsize('i'))/aspec_size)

    return N, nspec

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

def getCounts(booted_indices):
    bootnum, no_spectra = booted_indices.shape
    # # Does sorting helps with add.at?
    # booted_indices.sort(axis=1)
    # counts = np.zeros((bootnum, no_spectra), dtype=int)
    # for i, btdi in enumerate(booted_indices):
    #     np.add.at(counts[i], btdi, 1)
    counts = np.empty((bootnum, no_spectra), dtype=int)
    for b in range(bootnum):
        counts[b] = np.bincount(booted_indices[b], minlength=no_spectra)

    return counts

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
        
if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("BootDirectory", help="Directory where bootresults-N.dat are present.")
    parser.add_argument("--bootnum", default=1000, type=int, \
        help="Number of bootstrap resamples. Default: %(default)s")
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

    # Generating bootstrap realizations by file, constains the fluctuations
    # Generate one big boot array, each file gets their parts
    # We will need indices for each file
    nspec_total = int(0)
    indices     = np.empty(nbootfiles+1, dtype=int)
    Nbins, nspec= getNumbersfromBootfile(bootfiles[0])

    indices[0]  = 0
    for pe in range(nbootfiles):
        fname     = ospath_join(args.BootDirectory, f"bootresults-{pe}.dat")
        N1, nspec = getNumbersfromBootfile(fname)

        assert (N1 == Nbins)

        nspec_total  += nspec
        indices[pe+1] = nspec_total

    logging.info("There are %d spectra.", nspec_total)
    print(indices)

    # Generate bootstrap realizations through indexes
    RND            = np.random.default_rng(args.seed)
    booted_indices = RND.integers(low=0, high=nspec_total, size=(args.bootnum, nspec_total))
    boot_counts    = getCounts(booted_indices)
    logging.info(f"Generated boot indices.")

    # Allocate memory for matrices
    total_fisher   = np.zeros((args.bootnum, Nbins*Nbins))
    total_power_b4 = np.zeros((args.bootnum, Nbins))
    total_power    = np.zeros((args.bootnum, Nbins))

    # Set up progress tracker
    pcounter = Progress(nbootfiles)
    for pe in range(nbootfiles):
        fname = ospath_join(args.BootDirectory, f"bootresults-{pe}.dat")

        # Read fishers and powers
        fishers, powers = readBootFile(fname)
        this_counts     = boot_counts[:, indices[pe]:indices[pe+1]]

        total_fisher   += this_counts @ fishers
        total_power_b4 += this_counts @ powers

        pcounter.increase()

    logging.info("Calculating bootstrapped inverse Fisher and power...")
    for bi in range(args.bootnum):
        total_power[bi] = 0.5 * np.linalg.inv(total_fisher[bi].reshape(N,N)) @ total_power_b4[bi]

    # Save power to a file
    # Set up output file
    output_fname = ospath_join(args.BootDirectory, 
        "bootstrap-power-n%d-s%d.txt" % (args.bootnum, args.seed))
    np.savetxt(output_fname, total_power)
    logging.info("Power saved as ", output_fname)

    # If time allows, run many bootstraps and save its covariance
    # when save-cov passed
    if args.save_cov:
        bootstrap_cov = np.cov(total_power, rowvar=False)
        output_fname = ospath_join(args.BootDirectory, 
            "bootstrap-cov-n%d-s%d.txt" % (args.bootnum, args.seed))
        np.savetxt(output_fname, bootstrap_cov)
        logging.info("Covariance saved as ", output_fname)

