#!/usr/bin/env python
import argparse
from os.path import join as ospath_join

import numpy as np
from scipy.stats import binned_statistic
from astropy.table import Table

import qsotools.io as qio
import qsotools.fiducial as fid

def interpolate2Grid(v, f, padding = 100.):
    v1 = v[0] - padding
    v2 = v[-1] + padding
    dv = np.min(np.diff(v))

    Nv = int((v2-v1)/dv)
    new_varr = np.arange(Nv+1)*dv + v1 - dv/2

    interpF, _, _ = binned_statistic(v, f, statistic='sum', bins=new_varr)

    return interpF, dv

def binPowerSpectra(raw_k, raw_p, k_edges):
    binned_power,  _, binnumber = binned_statistic(raw_k, raw_p, statistic='sum', bins=k_edges)
    counts = np.bincount(binnumber, minlength=len(k_edges)+1)

    return binned_power, counts

def binCorrelations(raw_v, raw_c, r_edges):
    binned_corr,  _, binnumber = binned_statistic(raw_v, raw_c, statistic='sum', bins=r_edges)
    counts = np.bincount(binnumber, minlength=len(r_edges)+1)

    return binned_corr, counts

def getSpectographWindow2(k, Rint, dv):
    Rv = fid.LIGHT_SPEED / Rint / fid.ONE_SIGMA_2_FWHM
    x = k*dv/2/np.pi # numpy sinc convention multiplies x with pi
    
    W2k = np.exp(-(k*Rv)**2) * np.sinc(x)**2
    return W2k

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    parser.add_argument("--deconv-window", help="Deconvolve window function", \
        action="store_true")
    # parser.add_argument("--correlation", help="Compute correlation fn instead.")
    parser.add_argument("--dr", type=float, default=30.0)
    parser.add_argument("--nrbins", type=int, default=100)
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    file_list = open(config_qmle.qso_list, 'r')
    header = file_list.readline()

    power = np.zeros((config_qmle.z_n, config_qmle.k_bins.size))
    counts = np.zeros_like(power)

    r_edges = np.arange(args.nrbins+1) * args.dr
    r_bins  = (r_edges[1:] + r_edges[:-1]) / 2
    corr_fn = np.zeros((config_qmle.z_n, args.nrbins))
    counts_corr = np.zeros_like(corr_fn)

    for fl in file_list:
        print("Reading", fl.rstrip())
        f = ospath_join(config_qmle.qso_dir, fl.rstrip())
        bq = qio.BinaryQSO(f, 'r')

        z_med = bq.wave[int(bq.size/2)] / fid.LYA_WAVELENGTH - 1
        z_bin_no = int((z_med - config_qmle.z_0) / config_qmle.z_d)
        print("Median redshift:", z_med)

        if z_bin_no < 0 or z_bin_no > config_qmle.z_n-1:
            continue

        v_arr = fid.LIGHT_SPEED * np.log(bq.wave)
        delta_f, dv = interpolate2Grid(v_arr, bq.flux)

        # Compute & bin power
        p1d_f = np.abs(np.fft.rfft(delta_f) * dv)**2 / (dv*delta_f.size)
        this_k_arr = 2*np.pi*np.fft.rfftfreq(delta_f.size, dv)
        if args.deconv_window:
            p1d_f /= getSpectographWindow2(this_k_arr, bq.specres, bq.dv)

        # ignore k=0 mode
        p, c = binPowerSpectra(this_k_arr[1:], p1d_f[1:], config_qmle.k_edges)
        
        power[z_bin_no] += p
        counts[z_bin_no] += c[1:-1]

        # Compute and bin correlations
        new_varr = np.arange(delta_f.size)*dv
        corr1d_f = np.abs(np.fft.irfft(p1d_f)) / dv
        print(corr1d_f.shape)
        c, cc = binCorrelations(new_varr, corr1d_f, r_edges)

        corr_fn[z_bin_no] += c
        counts_corr[z_bin_no] += cc[1:-1]


    power /= counts
    corr_fn /= counts_corr

    p1d_filename = ospath_join(output_dir, output_base+"-p1d-fft-estimate.txt")
    corr_filename = ospath_join(output_dir, output_base+"-corr1d-fft-estimate.txt")

    zarr_repeated = np.repeat(config_qmle.z_bins, config_qmle.k_bins.size)
    karr_repeated = np.tile(config_qmle.k_bins, config_qmle.z_n)
    rarr_repeated = np.tile(r_bins, config_qmle.z_n)

    power_table = Table([zarr_repeated, karr_repeated, power.ravel()], names=('z', 'k', 'P1D'))
    power_table.write(p1d_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'k':'%.5e', 'P1D':'%.5e'}, overwrite=True)
    print("P1D saved as ", p1d_filename)

    corr_table = Table([zarr_repeated, rarr_repeated, corr_fn.ravel()], names=('z', 'r', 'Xi1D'))
    corr_table.write(corr_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'r':'%.1f', 'Xi1D':'%.5e'}, overwrite=True)
    print("Corr fn saved as ", corr_filename)



