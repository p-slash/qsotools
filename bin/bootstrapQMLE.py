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

# This function assumes spectra are organized s0/ s1/ .. folders
# and individual results are saved under s0/combined_Fp.fits
def qmleBootRun(booted_indices, qso_fname_list, N, inputdir, bootnum):
    total_fisher   = np.zeros((bootnum, N, N))
    total_power_b4 = np.zeros((bootnum, N))
    total_power    = np.zeros((bootnum, N))

    getSno = lambda x: int(re.search('/s(\d+)/desilite', x).group(1))
    getIDno= lambda x: int(re.search('_id(\d+)_', x).group(1))
    
    qso_fname_list.sort(key=getSno)
    qind = 0 # Stores the index in qso_fname_list for the loop

    for grno, sn_group in groupby(qso_fname_list, key=getSno):
        sn_list = list(sn_group)
        sn_list.sort(key=getIDno)

        fitsfile = fitsio.FITS(ospath_join(inputdir, "s%d"%grno, \
            "combined_Fp.fits.gz"), 'r')

        for sp in sn_list:
            this_id = getIDno(sp)
            data = fitsfile[this_id+1].read()[0]
            
            counts = np.count_nonzero(booted_indices==qind, axis=1)

            total_fisher   += data['fisher'][None,...]*counts[:, None, None]
            total_power_b4 += data['power'][None,:]*counts[:, None]
            qind+=1

        fitsfile.close()

        print("Results from s{:d}/combined_Fp.fits.gz are read and added.".format(grno))

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

    # # Create a dictionary, where keys are filenames and values are 
    # # numpy arrays of counts for bootstrap realizations.
    # bootstrap_counted_ind = np.array((no_spectra, args.bootnum))
    # for ind in range(no_spectra):
    #     bootstrap_counted_ind[ind] = np.count_nonzero(booted_indices==ind, axis=1)
    #     bootstrap_dict[qso_filename_list[ind]]=np.count_nonzero(booted_indices==ind, axis=1)
    # print("Dictionary that stores repetitions is calculated.")
    # print("Here's the first spectrum:", bootstrap_dict[qso_filename_list[0]])

    print("Running analysis...", flush=True)
    bootresult=qmleBootRun(booted_indices, qso_filename_list, N, config_qmle.qso_dir, args.bootnum)

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



