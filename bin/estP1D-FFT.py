#!/usr/bin/env python
import argparse
from os.path import join as ospath_join
import logging
from multiprocessing import Pool
from itertools import groupby

import numpy as np
from scipy.stats import binned_statistic
from astropy.table import Table

import qsotools.io as qio
import qsotools.fiducial as fid
from qsotools.specops import getSpectographWindow_k

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

def decomposePiccaFname(picca_fname):
    i1 = picca_fname.rfind('[')+1
    i2 = picca_fname.rfind(']')

    basefname = picca_fname[:i1-1]
    hdunum = int(picca_fname[i1:i2])

    return (basefname, hdunum)

class FFTEstimator(object):
    def __init__(self, args, config_qmle):
        self.args = args
        self.config_qmle = config_qmle

        self.power = np.zeros((self.config_qmle.z_n, self.config_qmle.k_bins.size))
        self.counts = np.zeros_like(power)
        self.mean_resolution = np.zeros(self.config_qmle.z_n)
        self.counts_meanreso = np.zeros_like(mean_resolution)

        self.r_edges = np.arange(args.nrbins+1) * args.dr
        self.corr_fn = np.zeros((self.config_qmle.z_n, args.nrbins))
        self.counts_corr = np.zeros_like(corr_fn)

    def getEstimates(self, qso):
        z_med = qso.wave[int(qso.size/2)] / fid.LYA_WAVELENGTH - 1
        z_bin_no = int((z_med - self.config_qmle.z_0) / self.config_qmle.z_d)

        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n-1:
            return

        v_arr = fid.LIGHT_SPEED * np.log(qso.wave)
        delta_f, dv = interpolate2Grid(v_arr, qso.flux)

        # Add to mean resolution
        self.mean_resolution[z_bin_no] += qso.specres
        self.counts_meanreso[z_bin_no] += 1

        # Compute & bin power
        p1d_f = np.abs(np.fft.rfft(delta_f) * dv)**2 / (dv*delta_f.size)

        if self.args.noise_realizations>0:
            pnoise = np.zeros_like(p1d_f)
            for _ in range(self.args.noise_realizations):
                delta_noise = np.random.default_rng().normal(0, qso.error)
                delta_noise, dv1 = interpolate2Grid(v_arr, delta_noise)
                pnoise += np.abs(np.fft.rfft(delta_noise) * dv1)**2 / (dv1*delta_noise.size)

            pnoise /= self.args.noise_realizations
            p1d_f -= pnoise

        this_k_arr = 2*np.pi*np.fft.rfftfreq(delta_f.size, dv)
        if self.args.deconv_window:
            p1d_f /= getSpectographWindow_k(this_k_arr, qso.specres, qso.dv)**2

        # ignore k=0 mode
        p, c = binPowerSpectra(this_k_arr[1:], p1d_f[1:], config_qmle.k_edges)
        
        self.power[z_bin_no] += p
        self.counts[z_bin_no] += c[1:-1]

        # Compute and bin correlations
        corr1d_f = np.abs(np.fft.irfft(p1d_f)) / dv
        new_varr = np.arange(corr1d_f.size)*dv
        c, cc = binCorrelations(new_varr, corr1d_f, self.r_edges)

        self.corr_fn[z_bin_no] += c
        self.counts_corr[z_bin_no] += cc[1:-1] 

    def __call__(self, fnames):
        if self.config_qmle.picca_input:
            decomp_list = [decomposePiccaFname(fl.rstrip()) for fl in fnames]
            decomp_list.sort(key=lambda x: x[0])
            for base, hdus in groupby(decomp_list, lambda x: x[0]):
                f = ospath_join(self.config_qmle.qso_dir, base)
                pfile = qio.PiccaFile(f, 'r', clobber=False)
                for hdu in hdus:
                    qso = pfile.readSpectrum(hdu[1])
                    self.getEstimates(qso)
                pfile.close()
        else:
            for fl in fnames:
                f = ospath_join(self.config_qmle.qso_dir, fl.rstrip())
                bq = qio.BinaryQSO(f, 'r')

                self.getEstimates(bq)

        return self.power, self.counts, self.corr_fn, self.counts_corr, \
            self.mean_resolution, self.counts_meanreso


if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    parser.add_argument("--deconv-window", help="Deconvolve window function", \
        action="store_true")
    # parser.add_argument("--correlation", help="Compute correlation fn instead.")
    parser.add_argument("--dr", type=float, default=30.0)
    parser.add_argument("--nrbins", type=int, default=100)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--noise-realizations", type=int, default=100)
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    file_list = open(config_qmle.qso_list, 'r')
    header = file_list.readline()

    power = np.zeros((config_qmle.z_n, config_qmle.k_bins.size))
    counts = np.zeros_like(power)

    mean_resolution = np.zeros(config_qmle.z_n)
    counts_meanreso = np.zeros_like(mean_resolution)

    r_edges = np.arange(args.nrbins+1) * args.dr
    r_bins  = (r_edges[1:] + r_edges[:-1]) / 2
    corr_fn = np.zeros((config_qmle.z_n, args.nrbins))
    counts_corr = np.zeros_like(corr_fn)

    fnames_spectra = file_list.readlines()
    nfchunk = int(len(fnames_spectra)/args.nproc)
    indices = np.arange(args.nproc+1)*nfchunk
    indices[-1] = len(fnames_spectra)
    fnames_spectra = [fnames_spectra[indices[i]:indices[i+1]] for i in range(args.nproc)]
    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(FFTEstimator(args, config_qmle), fnames_spectra)

        for p1, c1, corfn, cnts_crr, mean_res, counts_mreso in imap_it:
            power += p1
            counts += c1
            mean_resolution += mean_res
            counts_meanreso += counts_mreso
            corr_fn += corfn
            counts_corr += cnts_crr

    # Loop is done. Now average results
    power /= counts
    corr_fn /= counts_corr

    # Mean resolution
    mean_resolution /= counts_meanreso
    meanres_filename = ospath_join(output_dir, output_base+"-mean-resolution.txt")
    meanres_table = Table([config_qmle.z_bins, mean_resolution], names=('z', 'R'))
    meanres_table.write(meanres_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'R':'%d'}, overwrite=True)
    print("Mean R saved as ", meanres_filename)

    # Save power spectrum
    p1d_filename = ospath_join(output_dir, output_base+"-p1d-fft-estimate.txt")
    corr_filename = ospath_join(output_dir, output_base+"-corr1d-fft-estimate.txt")

    zarr_repeated = np.repeat(config_qmle.z_bins, config_qmle.k_bins.size)
    karr_repeated = np.tile(config_qmle.k_bins, config_qmle.z_n)

    power_table = Table([zarr_repeated, karr_repeated, power.ravel()], names=('z', 'k', 'P1D'))
    power_table.write(p1d_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'k':'%.5e', 'P1D':'%.5e'}, overwrite=True)
    print("P1D saved as ", p1d_filename)

    # Save correlation fn
    zarr_repeated = np.repeat(config_qmle.z_bins, r_bins.size)
    rarr_repeated = np.tile(r_bins, config_qmle.z_n)

    corr_table = Table([zarr_repeated, rarr_repeated, corr_fn.ravel()], names=('z', 'r', 'Xi1D'))
    corr_table.write(corr_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'r':'%.1f', 'Xi1D':'%.5e'}, overwrite=True)
    print("Corr fn saved as ", corr_filename)



