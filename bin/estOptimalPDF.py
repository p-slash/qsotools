#!/usr/bin/env python
import argparse
import logging
from multiprocessing import Pool

import numpy as np
from astropy.table import Table

import qsotools.io as qio
import qsotools.fiducial as fid
import qsotools.utils as qutil

class PDFEstimator(object):       
    def __init__(self, args, config_qmle, flux_edges, fiducial_corr_fn):
        self.args = args
        self.config_qmle = config_qmle
        self.flux_edges = flux_edges
        self.fiducial_corr_fn = fiducial_corr_fn

        self.nfbins = flux_edges.size-1
        self.z_edges = config_qmle.z_edges

        # self.minlength = (self.flux_edges.size+1) * (self.z_edges.size+1)

        self.flux_pdf = np.zeros(self.nfbins*self.config_qmle.z_n)
        self.fisher = np.zeros((self.flux_pdf.size, self.flux_pdf.size))

    def getInvCovariance(self, qso, z_arr):
        if self.args.smooth_noise_sigmaA > 0:
            qso.smoothNoise(sigma_A=self.args.smooth_noise_sigmaA)
        ivar = 1./qso.error**2
        ivar[~qso.mask] = 0

        v_arr = fid.LIGHT_SPEED * np.log(qso.wave/qso.wave[0])
        
        dv_matrix = v_arr[:, np.newaxis] - v_arr[np.newaxis, :]
        zij_matrix = np.sqrt(np.outer(1+z_arr, 1+z_arr))-1
        fiducial_signal = self.fiducial_corr_fn(zij_matrix, dv_matrix, grid=False)
        cinv = np.eye(qso.size)+np.diag(ivar)@fiducial_signal
        cinv = np.linalg.inv(cinv)*ivar

        return cinv

    def getEstimates(self, qso):
        z_arr = qso.wave/fid.LIGHT_SPEED-1
        z_med = z_arr[int(qso.size/2)]
        z_bin_no = int((z_med - self.config_qmle.z_edges[0]) / self.config_qmle.z_d)

        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n-1:
            return

        if self.args.convert2flux:
            qso.flux = (1+qso.flux) * fid.meanFluxFG08(z_arr)

        cinv = self.getInvCovariance(qso, z_arr)
        flux_idx = np.searchsorted(self.flux_edges, qso.flux)
        y = cinv@qso.flux
        i1 = z_bin_no * self.nfbins
        i2 = i1 + self.nfbins
        self.flux_pdf[i1:i2] += np.bincount(flux_idx, weights=y, minlength=self.nfbins+1)[1:-1]
        _2d_bin_idx = np.ravel(flux_idx[:, np.newaxis] + (self.nfbins+2) * flux_idx[np.newaxis, :])
        temp_cinv = np.bincount(_2d_bin_idx, minlength=(self.nfbins+2)**2)
        temp_cinv = temp_cinv.reshape(self.nfbins+2, self.nfbins+2)[1:-1, 1:-1]
        self.fisher[i1:i2, i1:i2] += temp_cinv
        # temp_cinv = np.empty((qso.size, self.nfbins))
        # for row in range(qso.size):
        #     temp_cinv[row] = np.bincount(flux_idx, weights=cinv[row, :], minlength=self.nfbins+1)[1:-1]
        # for col in range(qso.size):
        #     self.fisher[i1+col, i1:i2] += np.bincount(flux_idx, weights=temp_cinv[:, col], minlength=self.nfbins+1)[1:-1]

    def __call__(self, fname):
        if self.config_qmle.picca_input:
            base, hdus = fname
            f = f"{self.config_qmle.qso_dir}/{base}"
            pfile = qio.PiccaFile(f, 'r', clobber=False)

            for hdu in hdus:
                qso = pfile.readSpectrum(hdu)
                split_qsos = qso.split(self.config_qmle.z_edges)

                for qso in split_qsos:
                    self.getEstimates(qso)

            pfile.close()
        else:
            f = f"{self.config_qmle.qso_dir}/{fname.rstrip()}"
            bq = qio.BinaryQSO(f, 'r')

            self.getEstimates(bq)

        return self.flux_pdf, self.fisher


if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ConfigFile", help="Config file")

    parser.add_argument("--nsubsamples", type=int, default=100, \
        help="Number of subsamples if input is not Picca.")
    parser.add_argument("--f1", help="First flux bin", type=float, default=-0.2)
    parser.add_argument("--f2", help="Last flux bin", type=float, default=1.2)
    parser.add_argument("--df", help="Flux bin size", type=float, default=0.1)

    parser.add_argument("--smooth-noise-sigmaA", type=float, default=20.,
        help="Gaussian sigma in A to smooth pipeline noise estimates.")
    parser.add_argument("--dlambda", help="Wavelength dispersion", default=0.8)
    parser.add_argument("--covert2flux", help="Converts delta values to flux using FG08.",
        action="store_true")

    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--debug", help="Set logger to DEBUG level.", action="store_true")
    args = parser.parse_args()

    # Read Config file
    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    # Set up logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Set up flux bin edges
    nfbins = int((args.f2 - args.f1) / args.df)
    flux_edges = (np.arange(nfbins+1)-0.5) * args.df + args.f1
    flux_centers = (flux_edges[1:] + flux_edges[:-1])/2

    # Read file list file
    fnames_spectra = config_qmle.readFnameSpectra()

    # If files are in Picca format, decompose filename list into
    # Main file & hdus to read in that main file
    if config_qmle.picca_input:
        logging.info("Decomposing filenames to a list of (base, list(hdus)).")
        fnames_spectra = qutil.getPiccaFList(fnames_spectra)

    nfiles = len(fnames_spectra)

    # Calculate fiducial correlation function
    fiducial_corr_fn = fid.getLyaCorrFn(config_qmle.z_edges, args.dlambda)
    flux_pdf = np.zeros(nfbins*config_qmle.z_n)
    fisher = np.zeros((flux_pdf.size, flux_pdf.size))

    pcounter = qutil.Progress(nfiles) # Progress tracker
    logging.info(f"There are {nfiles} files.")
    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(
            PDFEstimator(args, config_qmle, flux_edges, fiducial_corr_fn),
            fnames_spectra
            )

        for _fpdf, _cinv in imap_it:
            flux_pdf += _fpdf
            fisher += _cinv

            pcounter.increase()

    for ii in range(fisher.size):
        if fisher[ii, ii] == 0:
            fisher[ii, ii]=1

    cov = np.linalg.inv(fisher)
    flux_pdf = cov@flux_pdf

    # Save flux pdf fn
    fname2save = f"{output_dir}/{output_base}-flux-pdf-estimate.txt"
    zarr_repeated = np.repeat(config_qmle.z_bins, nfbins)
    farr_repeated = np.tile(flux_centers, config_qmle.z_n)
    egauss = np.sqrt(cov.diagonal())

    corr_table = Table([zarr_repeated, farr_repeated, flux_pdf, egauss], \
        names=('z', 'F', 'FPDF', 'e_FPDF'))
    corr_table.write(fname2save, overwrite=True, \
        formats={'z':'%.5e', 'F':'%.5e', 'FPDF':'%.5e', 'e_FPDF':'%.5e'})
    logging.info(f"Flux PDF saved as {fname2save}")

    # Save Fisher
    fname2save = f"{output_dir}/{output_base}-fisher-flux-pdf-estimate.txt"
    np.savetxt(fname2save, fisher)

    logging.info("DONE!")
"""
    # Use subsampling to estimate covariance
    nsubsamples = nfiles if config_qmle.picca_input else args.nsubsamples
    # Set up subsampling class to store results
    reso_samples = qutil.SubsampleCov(config_qmle.z_n, nsubsamples, is_weighted=True)
    xi1d_samples = qutil.SubsampleCov(config_qmle.z_n*args.nrbins, nsubsamples, is_weighted=True)

    pcounter = qutil.Progress(nfiles) # Progress tracker
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
    meanres_filename = f"{output_dir}/{output_base}-mean-resolution.txt"
    meanres_table = Table([config_qmle.z_bins, mean_reso, err_reso], names=('z', 'R', 'e_R'))
    meanres_table.write(meanres_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'R':'%.1f', 'e_R':'%.1f'}, overwrite=True)
    logging.info(f"Mean R saved as {meanres_filename}")

    # Save correlation fn
    corr_filename = f"{output_dir}/{output_base}-corr1d-weighted-estimate.txt"
    zarr_repeated = np.repeat(config_qmle.z_bins, r_bins.size)
    rarr_repeated = np.tile(r_bins, config_qmle.z_n)

    err_xi1d = np.sqrt(cov_xi1d.diagonal())
    corr_table = Table([zarr_repeated, rarr_repeated, mean_xi1d, mean_xi1d_biascorr, err_xi1d], \
        names=('z', 'r', 'Xi1D', 'Xi1D-bcor', 'e_xi1d'))
    corr_table.write(corr_filename, format='ascii.fixed_width', overwrite=True, \
        formats={'z':'%.1f', 'r':'%.1f', 'Xi1D':'%.5e', 'Xi1D-bcor':'%.5e', 'e_xi1d':'%.5e'})
    logging.info(f"Corr fn saved as {corr_filename}")

    # Save covariance
    cov_filename = f"{output_dir}/{output_base}-covariance-xi1d-weighted-estimate.txt"
    np.savetxt(cov_filename, cov_xi1d)

    logging.info("DONE!")
"""






















