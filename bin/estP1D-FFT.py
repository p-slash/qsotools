#!/usr/bin/env python
import argparse
from os.path import join as ospath_join

import numpy as np
from scipy.stats import binned_statistic
from astropy.table import Table

import qsotools.io as qio
import qsotools.fiducial as fid

def interpolate2Grid(v, f, padding = 1000.):
    v1 = v[0] - padding
    v2 = v[-1] + padding
    dv = np.min(np.diff(v))

    Nv = int((v2-v1)/dv)
    new_varr = np.arange(Nv+1)*dv + v1 - dv/2

    interpF, _, _ = binned_statistic(v, f, statistic='sum', bins=new_varr)

    return interpF, dv

def binPowerSpectra(raw_p, raw_k, k_edges):
    binned_power,  _, binnumber = binned_statistic(raw_k, raw_p, statistic='sum', bins=k_edges)
    counts = np.bincount(binnumber, minlength=len(k_edges)+1)

    return binned_power, counts

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    k_edges, k_centers = fid.formBins(config_qmle.k_nlin, config_qmle.k_nlog, \
        config_qmle.k_dlin, config_qmle.k_dlog, config_qmle.k_0, config_qmle.k_ledge)

    file_list = open(config_qmle.qso_list, 'r')
    header = file_list.readline()

    power = np.zeros(len(k_centers))
    counts = np.zeros_like(power)

    for fl in file_list:
        f = ospath_join(config_qmle.qso_dir, fl.rstrip())
        bq = qio.BinaryQSO(f, 'r')
        bq.read()

        v_arr = fid.LIGHT_SPEED * np.log(bq.wave)
        delta_f, dv = interpolate2Grid(v_arr, bq.flux)

        p1d_f = np.abs(np.fft.rfft(delta_f))**2 * dv
        this_k_arr = np.fft.rfftfreq(delta_f.size, dv)

        p, c = binPowerSpectra(p1d_f, this_k_arr, k_edges)
        power += p
        counts += c

    power /= c

    p1d_filename = ospath_join(output_dir, output_base+"-p1d-fft-estimate.txt")

    power_table = Table([k_centers, power], names=('k', 'P1D'))

    power_table.write(p1d_filename, format='ascii.fixed_width', \
        formats={'k':'%.5e', 'P1D':'%.5e'}, overwrite=True)

     print("P1D saved as ", p1d_filename)



