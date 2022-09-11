#!/usr/bin/env python
import argparse
from os.path import join as ospath_join
import logging
from multiprocessing import Pool
from itertools import groupby

import numpy as np
from numba import jit
from astropy.table import Table

import qsotools.io as qio
import qsotools.fiducial as fid
from qsotools.utils import Progress, SubsampleCov

def decomposePiccaFname(picca_fname):
    i1 = picca_fname.rfind('[')+1
    i2 = picca_fname.rfind(']')

    basefname = picca_fname[:i1-1]
    hdunum = int(picca_fname[i1:i2])

    return (basefname, hdunum)

def _splitQSO(qso, z_edges):
    z = qso.wave/fid.LYA_WAVELENGTH-1
    sp_indx = np.searchsorted(z, z_edges)

    assert len(np.hsplit(z, sp_indx)) == len(sp_indx)+1

    # logging.debug(", ".join(np.hsplit(qso.wave, sp_indx)))
    wave_chunks  = [x for x in np.hsplit(qso.wave, sp_indx)[1:-1]  if 0 not in x.shape]
    flux_chunks  = [x for x in np.hsplit(qso.flux, sp_indx)[1:-1]  if 0 not in x.shape]
    error_chunks = [x for x in np.hsplit(qso.error, sp_indx)[1:-1] if 0 not in x.shape]
    reso_chunks  = [x for x in np.hsplit(qso.reso_kms, sp_indx)[1:-1] if 0 not in x.shape]
    # logging.debug(", ".join(wave_chunks))

    split_qsos =[]
    for i in range(len(wave_chunks)):
        wave = wave_chunks[i]
        flux = flux_chunks[i]
        error= error_chunks[i]
        reso_kms = reso_chunks[i]

        tmp_qso = qio.Spectrum(wave, flux, error, qso.z_qso, \
            qso.specres, qso.dv, qso.coord, reso_kms)

        if tmp_qso.s2n > 0 and wave.size > 10:
            split_qsos.append(tmp_qso)

    return split_qsos

@jit("i8(f8[:], i8, f8)", nopython=True)
def _findVMaxj(arr, j1, rmax):
    for j in range(j1, arr.size):
        if arr[j] > rmax:
            return j

    return arr.size

@jit("f8[:](f8[:], f8[:], f8[:], f8[:])", nopython=True)
def _getXi1D(v_arr, flux, ivar, r_edges):
    # 1d array to store results
    # first N : Xi_1d , second N : Weights
    Nbins = r_edges.size-1
    bin_res = np.zeros(2*Nbins)

    # Compute and bin correlations
    last_max_j = int(0)
    for i in range(v_arr.size):
        if ivar[i] < 1e-4:
            continue

        last_max_j = _findVMaxj(v_arr, last_max_j, r_edges[-1]+v_arr[i])
        vrange = slice(i, last_max_j)

        vdiff = v_arr[vrange] - v_arr[i]

        sub_xi1d = flux[i]*flux[vrange]
        sub_w1d  = ivar[i]*ivar[vrange]

        sp_indx = np.searchsorted(r_edges, vdiff)
        bin_res[:Nbins] += np.bincount(sp_indx, weights=sub_xi1d, minlength=r_edges.size+1)[1:-1]
        bin_res[Nbins:] += np.bincount(sp_indx, weights=sub_w1d, minlength=r_edges.size+1)[1:-1]

    return bin_res

class Xi1DEstimator(object):
    def __init__(self, args, config_qmle):
        self.args = args
        self.config_qmle = config_qmle

        self.r_edges = np.arange(args.nrbins+1) * args.dr
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
        z_bin_no = int((z_med - self.config_qmle.z_edges[0]) / self.config_qmle.z_d)

        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n-1:
            return

        v_arr = fid.LIGHT_SPEED * np.log(qso.wave/qso.wave[0])

        # Add to mean resolution
        w = qso.reso_kms > 0
        self.mean_resolution[z_bin_no] += np.sum(qso.reso_kms[w])
        self.counts_meanreso[z_bin_no] += np.sum(w)

        ivar = self.getIVAR(qso)
        # Weighted deltas
        wflux = qso.flux * ivar

        binres = _getXi1D(v_arr, wflux, ivar, self.r_edges)
        self.xi1d[z_bin_no]   += binres[:self.args.nrbins]
        self.counts[z_bin_no] += binres[self.args.nrbins:]

    def __call__(self, fname):
        if self.config_qmle.picca_input:
            base, hdus = fname
            f = ospath_join(self.config_qmle.qso_dir, base)
            pfile = qio.PiccaFile(f, 'r', clobber=False)

            for hdu in hdus:
                qso = pfile.readSpectrum(hdu)
                split_qsos = _splitQSO(qso, self.config_qmle.z_edges)

                for qso in split_qsos:
                    self.getEstimates(qso)

            pfile.close()
        else:
            f = ospath_join(self.config_qmle.qso_dir, fname.rstrip())
            bq = qio.BinaryQSO(f, 'r')

            self.getEstimates(bq)

        return self.xi1d, self.counts, self.mean_resolution, self.counts_meanreso


if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")

    parser.add_argument("--nsubsamples", type=int, default=100, \
        help="Number of subsamples if input is not Picca.")
    parser.add_argument("--dr", help="Default: %(default)s km/s", type=float, default=70.0)
    parser.add_argument("--nrbins", help="Default: %(default)s", type=int, default=60)
    parser.add_argument("--smooth-noise-sigmaA", type=float, default=20.,
        help="Gaussian sigma in A to smooth pipeline noise estimates. Default: %(default)s A")
    # parser.add_argument("--project-out", action="store_true", \
    #     help="Projects out mean and slope modes.")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--debug", help="Set logger to DEBUG level.", action="store_true")
    args = parser.parse_args()

    # Read Config file
    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    # Set up logger
    logging.basicConfig(filename=ospath_join(output_dir, f'est-xi1d.log'), \
        level=logging.DEBUG if args.debug else logging.INFO)

    # Set up velocity bin edges
    r_edges = np.arange(args.nrbins+1) * args.dr
    r_bins  = (r_edges[1:] + r_edges[:-1]) / 2

    # Read file list file
    file_list = open(config_qmle.qso_list, 'r')
    header = file_list.readline() # First line: Number of spectra to read
    fnames_spectra = file_list.readlines()
    fnames_spectra = fnames_spectra[:int(header)] # Read only first N spectra

    # If files are in Picca format, decompose filename list into
    # Main file & hdus to read in that main file
    if config_qmle.picca_input:
        logging.info("Decomposing filenames to a list of (base, list(hdus)).")
        decomp_list = [decomposePiccaFname(fl.rstrip()) for fl in fnames_spectra]
        decomp_list.sort(key=lambda x: x[0])

        new_fnames = []
        for base, hdus in groupby(decomp_list, lambda x: x[0]):
            new_fnames.append((base, list(map(lambda x: x[1], hdus))))

        fnames_spectra = new_fnames

    nfiles = len(fnames_spectra)
    # Use subsampling to estimate covariance
    nsubsamples = nfiles if config_qmle.picca_input else args.nsubsamples
    # Set up subsampling class to store results
    reso_samples = SubsampleCov(config_qmle.z_n, nsubsamples, is_weighted=True)
    xi1d_samples = SubsampleCov(config_qmle.z_n*args.nrbins, nsubsamples, is_weighted=True)

    pcounter = Progress(nfiles) # Progress tracker
    logging.info(f"There are {nfiles} files.")
    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(Xi1DEstimator(args, config_qmle), fnames_spectra)

        for corfn, cnts_crr, mean_res, counts_mreso in imap_it:
            reso_samples.addMeasurement(mean_res, counts_mreso)
            xi1d_samples.addMeasurement(corfn.ravel(), cnts_crr.ravel())

            pcounter.increase()

    # Loop is done. Now average results
    mean_xi1d, cov_xi1d = xi1d_samples.getMeanNCov()
    mean_xi1d_biascorr, _ = xi1d_samples.getMeanNCov(bias_correct=True)
    mean_reso, cov_reso = reso_samples.getMeanNCov()

    # Mean resolution
    err_reso = np.sqrt(cov_reso.diagonal())
    meanres_filename = ospath_join(output_dir, output_base+"-mean-resolution.txt")
    meanres_table = Table([config_qmle.z_bins, mean_reso, err_reso], names=('z', 'R', 'e_R'))
    meanres_table.write(meanres_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'R':'%.1f', 'e_R':'%.1f'}, overwrite=True)
    logging.info(f"Mean R saved as {meanres_filename}")

    # Save correlation fn
    corr_filename = ospath_join(output_dir, output_base+"-corr1d-weighted-estimate.txt")
    zarr_repeated = np.repeat(config_qmle.z_bins, r_bins.size)
    rarr_repeated = np.tile(r_bins, config_qmle.z_n)

    err_xi1d = np.sqrt(cov_xi1d.diagonal())
    corr_table = Table([zarr_repeated, rarr_repeated, mean_xi1d, mean_xi1d_biascorr, err_xi1d], \
        names=('z', 'r', 'Xi1D', 'Xi1D-bcor', 'e_xi1d'))
    corr_table.write(corr_filename, format='ascii.fixed_width', overwrite=True, \
        formats={'z':'%.1f', 'r':'%.1f', 'Xi1D':'%.5e', 'Xi1D-bcor':'%.5e', 'e_xi1d':'%.5e'})
    logging.info(f"Corr fn saved as {corr_filename}")

    # Save covariance
    cov_filename = ospath_join(output_dir, output_base+"-covariance-xi1d-weighted-estimate.txt")
    np.savetxt(cov_filename, cov_xi1d)

    logging.info("DONE!")























