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
from qsotools.utils import Progress

def decomposePiccaFname(picca_fname):
    i1 = picca_fname.rfind('[')+1
    i2 = picca_fname.rfind(']')

    basefname = picca_fname[:i1-1]
    hdunum = int(picca_fname[i1:i2])

    return (basefname, hdunum)

class Xi1DEstimator(object):
    def __init__(self, args, config_qmle):
        self.args = args
        self.config_qmle = config_qmle

        self.r_edges = np.arange(args.nrbins+1) * args.dr
        self.rmax = self.r_edges[-1]
        self.xi1d = np.zeros((self.config_qmle.z_n, args.nrbins))
        self.counts = np.zeros_like(self.xi1d)
        self.mean_resolution = np.zeros(self.config_qmle.z_n)
        self.counts_meanreso = np.zeros_like(self.mean_resolution)

    def getIVAR(self, qso):
        # Smooth qso.error
        if args.smooth_noise_sigmaA > 0:
            qso.smoothNoise(sigma_A=args.smooth_noise_sigmaA)

        # Add LSS to qso.error**2
        qso.addLyaFlucErrors(on_flux=False)

        ivar = 1./qso.error**2
        ivar[~qso.mask] = 0

        return ivar

    def getEstimates(self, qso):
        z_med = qso.wave[int(qso.size/2)] / fid.LYA_WAVELENGTH - 1
        z_bin_no = int((z_med - self.config_qmle.z_0) / self.config_qmle.z_d)

        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n-1:
            return

        v_arr = fid.LIGHT_SPEED * np.log(qso.wave)

        # Add to mean resolution
        self.mean_resolution[z_bin_no] += qso.specres
        self.counts_meanreso[z_bin_no] += 1

        ivar = self.getIVAR(qso)
        # Weighted deltas
        qso.flux *= ivar

        # Compute and bin correlations
        for i in range(qso.size):
            vrange = (v_arr[i:]-v_arr[i]) < self.rmax

            if not np.any(vrange):
                break

            vdiff = v_arr[i:][vrange] - v_arr[i]

            sub_xi1d = qso.flux[i]*qso.flux[i:][vrange]
            sub_w1d  = ivar[i]*ivar[i:][vrange]

            binned_xi1d = binned_statistic(vdiff, sub_xi1d, statistic='sum', bins=self.r_edges)[0]
            binned_w1d  = binned_statistic(vdiff, sub_w1d, statistic='sum', bins=self.r_edges)[0]

            self.xi1d[z_bin_no] += binned_xi1d
            self.counts[z_bin_no] += binned_w1d

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

        return self.xi1d, self.counts, self.mean_resolution, self.counts_meanreso


if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")

    parser.add_argument("--dr", help="Default: %(default)s km/s", type=float, default=30.0)
    parser.add_argument("--nrbins", help="Default: %(default)s", type=int, default=100)
    parser.add_argument("--smooth-noise-sigmaA", type=float, default=20.,
        help="Gaussian sigma in A to smooth pipeline noise estimates. Default: %(default)s A")
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    logging.basicConfig(filename=ospath_join(output_dir, f'est-xi1d.log'), \
        level=logging.INFO)

    file_list = open(config_qmle.qso_list, 'r')
    header = file_list.readline()

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

    pcounter = Progress(len(fnames_spectra))
    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(Xi1DEstimator(args, config_qmle), fnames_spectra)

        for corfn, cnts_crr, mean_res, counts_mreso in imap_it:
            mean_resolution += mean_res
            counts_meanreso += counts_mreso
            corr_fn += corfn
            counts_corr += cnts_crr

            pcounter.increase()

    # Loop is done. Now average results
    corr_fn /= counts_corr

    # Mean resolution
    mean_resolution /= counts_meanreso
    meanres_filename = ospath_join(output_dir, output_base+"-mean-resolution.txt")
    meanres_table = Table([config_qmle.z_bins, mean_resolution], names=('z', 'R'))
    meanres_table.write(meanres_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'R':'%d'}, overwrite=True)
    logging.info(f"Mean R saved as {meanres_filename}")

    # Save correlation fn
    corr_filename = ospath_join(output_dir, output_base+"-corr1d-weighted-estimate.txt")
    zarr_repeated = np.repeat(config_qmle.z_bins, r_bins.size)
    rarr_repeated = np.tile(r_bins, config_qmle.z_n)

    corr_table = Table([zarr_repeated, rarr_repeated, corr_fn.ravel()], names=('z', 'r', 'Xi1D'))
    corr_table.write(corr_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'r':'%.1f', 'Xi1D':'%.5e'}, overwrite=True)
    logging.info(f"Corr fn saved as {corr_filename}")

    logging.info("DONE!")


