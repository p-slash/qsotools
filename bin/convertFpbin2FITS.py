#!/usr/bin/env python
import argparse
import struct
import fitsio
from os.path import join as ospath_join

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

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    N = (config_qmle.k_nlin + config_qmle.k_nlog) * config_qmle.z_n
    print("Config file is read.")

    # Read qso filenames into a list, then convert to numpy array
    with open(config_qmle.qso_list, 'r') as file_qsolist:
        header = file_qsolist.readline()
        qso_filename_list = [ospath_join(config_qmle.qso_dir, x.rstrip()[:-4]+"_Fp.bin") \
            for x in file_qsolist]

    no_spectra = len(qso_filename_list)
    dtype_fp = np.dtype([('fisher', np.double, (N,N)), ('power', np.double, N)])
    data = np.zeros(1, dtype=dtype_fp)

    fits = fitsio.FITS(ospath_join(config_qmle.qso_dir), "combined_Fp.fits.gz")

    for f in qso_filename_list:
        fisher, power = readFPBinFile(f)
        data['fisher'] = fisher
        data['power'] = power
        fits.write(data)

    fits.close()


















