#!/usr/bin/env python
import argparse
from os.path import join as ospath_join
from itertools import groupby

import struct
import numpy as np
import fitsio
import re

import qsotools.io as qio

def readFPBinFile(fname):
    with open(fname, "rb") as fpbin:
        N = int(struct.unpack('i', fpbin.read(struct.calcsize('i')))[0])
        
        fisher_fmt = 'd'*N*N
        fisher = struct.unpack(fisher_fmt, fpbin.read(struct.calcsize(fisher_fmt)))
        fisher = np.array(fisher, dtype=np.double)
        fisher = fisher.reshape((N, N))

        power_fmt = 'd'*N
        power = struct.unpack(power_fmt, fpbin.read(struct.calcsize(power_fmt)))
        power = np.array(power, dtype=np.double)

    return fisher, power

def getCounts(booted_indices, bootnum, no_spectra):
    # Does sorting helps with add.at?
    booted_indices.sort(axis=1)
    counts = np.zeros((bootnum, no_spectra), dtype=np.int)
    for i, btdi in enumerate(booted_indices):
        np.add.at(counts[i], btdi, 1)

    return np.transpose(counts)

# This function assumes spectra are organized s0/ s1/ .. folders
# and individual results are saved under s0/combined_Fp.fits
def qmleBootRun(booted_indices, qso_fname_list, N, inputdir, bootnum, fp_file):
    total_fisher   = np.zeros((bootnum, N, N))
    total_power_b4 = np.zeros((bootnum, N))
    total_power    = np.zeros((bootnum, N))

    sno_regex= re.compile('/s(\d+)/desilite')
    getSno = lambda x: int(sno_regex.search(x).group(1))
    # id_regex = re.compile('_id(\d+)_')
    # getIDno= lambda x: int(id_regex.search(x).group(1))
    
    qso_fname_list.sort(key=getSno) # Sort for groupby
    no_spectra = len(qso_filename_list)

    # counts shape (no_spectra, bootnum)
    print("Getting repetitions...", flush=True)
    counts = getCounts(booted_indices, bootnum, no_spectra)

    qind = 0 # Stores the index in qso_fname_list for the loop

    for grno, sn_group in groupby(qso_fname_list, key=getSno):
        fitspath = ospath_join(inputdir, "s%d"%grno, fp_file)
        print("Reading {:s}...".format(fitspath), flush=True)

        with fitsio.FITS(fitspath) as fitsfile:
            for hdu in fitsfile[1:]:
                ci = counts[qind]
                data = hdu.read()
                
                total_fisher   += data['fisher']*ci[:, None, None]
                total_power_b4 += data['power']*ci[:, None]
                
                qind+=1
                if qind%4000==0:
                    print("Progress: {:4.1f}%".format(100*qind/no_spectra), flush=True)

        print("Results from {:s} are read and added.".format(fitspath, grno), flush=True)

    print("Calculating bootstrapped inverse Fisher and power...", flush=True)
    for bi in range(bootnum):
        inv_total_fisher = np.linalg.inv(total_fisher[bi]) 
        total_power[bi] = 0.5 * inv_total_fisher @ total_power_b4[bi]
    
    return total_power

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    parser.add_argument("--bootnum", default=1000, type=int, \
        help="Number of bootstrap resamples. Default: %(default)s")
    parser.add_argument("--fp-file", default="combined_Fp.fits")
    parser.add_argument("--seed", default=3422, type=int)
    parser.add_argument("--save-cov", action="store_true")
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    N = (config_qmle.k_nlin + config_qmle.k_nlog) * config_qmle.z_n
    print("Config file is read.")

    # Read qso filenames into a list, then convert to numpy array
    with open(config_qmle.qso_list, 'r') as file_qsolist:
        header = file_qsolist.readline()
        qso_filename_list = [ospath_join(config_qmle.qso_dir, x.rstrip()) \
            for x in file_qsolist]

    no_spectra = len(qso_filename_list)
    print("Filenames of {:d} spectra are stored.".format(no_spectra))

    # Generate random indices as bootstrap
    RND = np.random.RandomState(args.seed)
    booted_indices = RND.randint(no_spectra, size=(args.bootnum, no_spectra))
    print("{:d} bootstrap realisations are generated.".format(args.bootnum))
    print("Here's the first realisation:", booted_indices[0], flush=True)
    print("Here's the repetitions for index 0:", \
        np.count_nonzero(booted_indices==0, axis=1), flush=True)

    print("Running analysis...", flush=True)
    bootresult=qmleBootRun(booted_indices, qso_filename_list, N, config_qmle.qso_dir, \
        args.bootnum, args.fp_file)

    # Save power to a file
    power_filename = ospath_join(output_dir, output_base \
        +"-bootstrap-power-n%d-s%d.txt" % (args.bootnum, args.seed))
    np.savetxt(power_filename, bootresult)
    print("Power saved as ", power_filename)

    # If time allows, run many bootstraps and save its covariance
    # when save-cov passed
    if args.save_cov:
        bootstrap_cov = np.cov(bootresult, rowvar=False)
        cov_filename = ospath_join(output_dir, output_base \
            +"-bootstrap-cov-n%d-s%d.txt" % (args.bootnum, args.seed))
        np.savetxt(cov_filename, bootstrap_cov)
        print("Covariance saved as ", cov_filename)



