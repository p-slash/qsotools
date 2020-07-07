#!/usr/bin/env python
import argparse
from os.path import join as ospath_join
from collections import Counter
from itertools import groupby

import struct
import numpy as np
import fitsio
import re

from astropy.utils import NumpyRNGContext
from astropy.stats import bootstrap

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
def qmleBootRun(qso_fname_list, N, inputdir):
    total_fisher   = np.zeros((N,N))
    total_power_b4 = np.zeros(N)

    qso_fname_list.sort()
    getSno = lambda x: int(re.search('/s(\d+)/desilite', x).group(1))
    getIDno= lambda x: int(re.search('_id(\d+)_', x).group(1))

    for grno, sn_group in groupby(qso_fname_list, key=getSno):
        sn_list = list(sn_group)
        sn_list.sort(key=getIDno)
        c = Counter(sn_list)

        fitsfile = fitsio.FITS(ospath_join(inputdir, "s%d"%grno, \
            "combined_Fp.fits"), 'r')

        for elem in c:
            this_id = getIDno(elem)
            count = c[elem]

            data = fitsfile[this_id+1].read()[0]

            total_fisher   += data['fisher']*count
            total_power_b4 += data['power']*count

        fitsfile.close()

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
    parser.add_argument("--save-cov", action="store_true")
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    N = (config_qmle.k_nlin + config_qmle.k_nlog) * config_qmle.z_n

    # Read qso filenames into a list, then convert to numpy array
    with open(config_qmle.qso_list, 'r') as file_qsolist:
        header = file_qsolist.readline()
        qso_filename_list = np.array([ospath_join(config_qmle.qso_dir, x.rstrip()) \
            for x in file_qsolist])

    # Set up bootstrap statistics function defined above
    qmleBootRunN = lambda x: qmleBootRun(x, N, config_qmle.qso_dir)

    # Bootstrap to get a new power estimate
    with NumpyRNGContext(args.seed):
        bootresult = bootstrap(qso_filename_list, bootnum=args.bootnum, \
            bootfunc=qmleBootRunN)

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



