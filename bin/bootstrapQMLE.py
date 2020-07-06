#!/usr/bin/env python
import argparse
from os.path import join as ospath_join

import struct
import numpy as np

from astropy.utils import NumpyRNGContext
from astropy.stats import bootstrap

import qsotools.io as qio

def readFPBinFile(fname):
    fpbin = open(fname, "rb")
    N = int(struct.unpack('i', fpbin.read(struct.calcsize('i')))[0])
    
    fisher_fmt = 'd'*N*N
    fisher = struct.unpack(fisher_fmt, fpbin.read(struct.calcsize(fisher_fmt)))
    fisher = np.array(fisher, dtype=np.double)
    fisher = fisher.reshape((N, N))

    power_fmt = 'd'*N
    power = struct.unpack(power_fmt, fpbin.read(struct.calcsize(power_fmt)))
    power = np.array(power, dtype=np.double)
    
    fpbin.close()

    return fisher, power

def qmleBootRun(qso_fname_list, N):
    total_fisher   = np.zeros((N,N))
    total_power_b4 = np.zeros(N)

    for qso_fname in qso_fname_list:
        fp_result_fname = qso_fname.replace(".dat", "_Fp.bin")

        fisher, power = readFPBinFile(fp_result_fname)
        
        total_fisher   += fisher
        total_power_b4 += power

    inv_total_fisher = np.linalg.inv(total_fisher) 
    total_power = 0.5 * inv_total_fisher @ total_power_b4
    
    return total_power


if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    parser.add_argument("--bootnum", default=1000, type=int, \
        help="Number of bootstrap resamples. Default: %(default)s")
    parser.add_argument("--seed", default=3422, type=int)
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)

    N = (config_qmle.k_nlin + config_qmle.k_nlog) * config_qmle.z_n

    file_qsolist = open(config_qmle.qso_list, 'r')
    header = file_qsolist.readline()
    qso_filename_list = file_qsolist.readlines()
    qso_filename_list = [ospath_join(config_qmle.qso_dir, x.rstrip()) \
        for x in qso_filename_list]

    qmleBootRunN = lambda x: qmleBootRun(x, N)

    with NumpyRNGContext(args.seed):
        bootresult = bootstrap(qso_filename_list, bootnum=args.bootnum, \
            bootfunc=qmleBootRunN)

    bootstrap_cov = np.cov(bootresult, rowvar=False)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']
    cov_filename = ospath_join(output_dir, output_base+"-bootstrap-cov.txt")
    np.savetxt(cov_filename, bootstrap_cov)
    print("Saves as ", cov_filename)



