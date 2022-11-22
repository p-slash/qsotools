#!/usr/bin/env python
import argparse
import logging

import numpy as np
import cupy
from astropy.table import Table
from mpi4py import MPI

import qsotools.io as qio
import qsotools.fiducial as fid
import qsotools.utils as qutil

class cpInterp2d(object):
    def __init__(self, x_cpu, y_cpu, f_cpu):
        assert np.allclose(np.diff(x_cpu), x_cpu[1]-x_cpu[0])
        assert np.allclose(np.diff(y_cpu), y_cpu[1]-y_cpu[0])
        assert f_cpu.shape == (x_cpu.size, y_cpu.size)

        self.x_gpu = cupy.asarray(x_cpu)
        self.y_gpu = cupy.asarray(y_cpu)
        self.f_gpu = cupy.asarray(f_cpu)
        self.dx = self.x_gpu[1]-self.x_gpu[0]
        self.dy = self.y_gpu[1]-self.y_gpu[0]

    def __call__(self, x_g, y_g):
        # Limit
        x_g[x_g<self.x_gpu[0]] = self.x_gpu[0]
        x_g[x_g>=self.x_gpu[-1]] = self.x_gpu[-1]-1e-6
        y_g[y_g<self.y_gpu[0]] = self.y_gpu[0]
        y_g[y_g>=self.y_gpu[-1]] = self.y_gpu[-1]-1e-6

        xx = (x_g-self.x_gpu[0])/self.dx
        xi = (xx).astype(np.int64)
        dxi = xx - xi

        yy = (y_g-self.y_gpu[0])/self.dy
        yi = (yy).astype(np.int64)
        dyi = yy - yi

        result = self.f_gpu[xi, yi] * (1-dxi) * (1-dyi)\
            + self.f_gpu[xi+1, yi] * (dxi) * (1-dyi)\
            + self.f_gpu[xi, yi+1] * (1-dxi) * (dyi)\
            + self.f_gpu[xi+1, yi+1] * (dxi) * (dyi)

        return result

def cp_meanFluxFG08(z):
    tau = 0.001845 * cupy.power(1. + z, 3.924)

    return cupy.exp(-tau)

def logging_mpi(msg, mpi_rank):
    if mpi_rank == 0:
        logging.info(msg)

def balance_load(pc_flist, mpi_size, mpi_rank):
    pc_flist.sort(key=lambda x: len(x[1])) # Ascending order
    number_of_spectra = np.zeros(mpi_size, dtype=int)
    local_queue = []
    for x in reversed(pc_flist):
        min_idx = np.argmin(number_of_spectra)
        number_of_spectra[min_idx] += len(x[1])

        if min_idx == mpi_rank:
            local_queue.append(x)

    return local_queue

class PDFEstimator(object):       
    def __init__(self, args, config_qmle, flux_edges_gpu, fiducial_corr_fn):
        self.args = args
        self.config_qmle = config_qmle
        self.flux_edges_gpu = flux_edges_gpu
        self.fiducial_corr_fn = fiducial_corr_fn

        self.nfbins = flux_edges_gpu.size-1
        self.z_edges = config_qmle.z_edges

        self.flux_pdf = cupy.zeros(self.nfbins*self.config_qmle.z_n)
        self.fisher = cupy.zeros((self.flux_pdf.size, self.flux_pdf.size))

    def getInvCovariance(self, w_gpu, z_gpu, e_gpu):
        v_gpu = w_gpu/w_gpu[0]
        v_gpu = fid.LIGHT_SPEED * cupy.log(v_gpu)
        dv_matrix = v_gpu[:, cupy.newaxis] - v_gpu[cupy.newaxis, :]
        zij_matrix = cupy.sqrt(cupy.outer(1+z_gpu, 1+z_gpu))-1

        cinv_gpu = self.fiducial_corr_fn(zij_matrix, dv_matrix)

        _di_idx = cupy.diag_indices(v_gpu.size)
        cinv_gpu[_di_idx] += e_gpu**2
        cinv_gpu = cupy.linalg.inv(cinv_gpu)

        return cinv_gpu

    def getEstimates(self, qso):
        if self.args.smooth_noise_sigmaA > 0:
            qso.smoothNoise(sigma_A=self.args.smooth_noise_sigmaA)

        w_gpu = cupy.asarray(qso.wave)
        z_gpu = w_gpu/fid.LYA_WAVELENGTH-1
        f_gpu = cupy.asarray(qso.flux)
        e_gpu = cupy.asarray(qso.error)

        # z_arr = qso.wave/fid.LYA_WAVELENGTH-1
        z_med = z_gpu[int(qso.size/2)]
        z_bin_no = int((z_med - self.config_qmle.z_edges[0]) / self.config_qmle.z_d)

        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n-1:
            logging.debug(f"Skipping z_med={z_med:.2f}")
            return

        if self.args.convert2flux:
            mf = cp_meanFluxFG08(z_gpu)
            f_gpu = (1+f_gpu) * mf
            e_gpu *= mf

        cinv_gpu = self.getInvCovariance(w_gpu, z_gpu, e_gpu)
        flux_idx_gpu = cupy.searchsorted(self.flux_edges_gpu, f_gpu)

        y = cinv_gpu.dot(f_gpu)
        i1 = z_bin_no * self.nfbins
        i2 = i1 + self.nfbins
        self.flux_pdf[i1:i2] += cupy.bincount(flux_idx_gpu, weights=y, minlength=self.nfbins+2)[1:-1]

        _2d_bin_idx = cupy.ravel(
            flux_idx_gpu[:, cupy.newaxis]
            + (self.nfbins+2) * flux_idx_gpu[cupy.newaxis, :]
        )
        temp_cinv = cupy.bincount(_2d_bin_idx, weights=cinv_gpu.ravel(), minlength=(self.nfbins+2)**2)
        temp_cinv = temp_cinv.reshape(self.nfbins+2, self.nfbins+2)[1:-1, 1:-1]
        self.fisher[i1:i2, i1:i2] += temp_cinv

    def __call__(self, fname):
        if self.config_qmle.picca_input:
            base, hdus = fname
            f = f"{self.config_qmle.qso_dir}/{base}"
            pfile = qio.PiccaFile(f, 'r', clobber=False)

            for hdu in hdus:
                qso = pfile.readSpectrum(hdu)
                split_qsos = qso.split(self.config_qmle.z_edges, self.args.min_nopix)

                for qso in split_qsos:
                    self.getEstimates(qso)

            pfile.close()
        else:
            f = f"{self.config_qmle.qso_dir}/{fname.rstrip()}"
            bq = qio.BinaryQSO(f, 'r')

            self.getEstimates(bq)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

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
    parser.add_argument("--dlambda", help="Wavelength dispersion", type=float, default=0.8)
    parser.add_argument("--convert2flux", help="Converts delta values to flux using FG08.",
        action="store_true")
    parser.add_argument("--min-nopix", help="Minimum number of pixels in chunk", type=int,
        default=20)

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
    flux_edges_gpu = (cupy.arange(nfbins+1)-0.5) * args.df + args.f1
    _cdev = flux_edges_gpu.device
    logging.debug(f"mpi: {mpi_rank} -- gpu: {_cdev.id} - bus {_cdev.pci_bus_id}")

    # Read file list file
    fnames_spectra = config_qmle.readFnameSpectra()

    # If files are in Picca format, decompose filename list into
    # Main file & hdus to read in that main file
    if config_qmle.picca_input:
        logging_mpi("Decomposing filenames to a list of (base, list(hdus)).", mpi_rank)
        fnames_spectra = qutil.getPiccaFList(fnames_spectra)

    nfiles = len(fnames_spectra)
    logging_mpi(f"There are {nfiles} files.", mpi_rank)
    local_queue = balance_load(fnames_spectra, mpi_size, mpi_rank)

    # Calculate fiducial correlation function
    logging_mpi("Calculating fiducial correlation function.", mpi_rank)
    z, v, xi1d = fid.getLyaCorrFn(config_qmle.z_edges, args.dlambda)
    fiducial_corr_fn = cpInterp2d(z, v, xi1d)

    logging_mpi("Running PDF estimator.", mpi_rank)
    pdf_estimator = PDFEstimator(args, config_qmle, flux_edges_gpu, fiducial_corr_fn)
    for fname in local_queue:
        pdf_estimator(fname)

    logging.info(f"Finished mpi:{mpi_rank}")

    flux_edges_cpu = cupy.asnumpy(flux_edges_gpu)
    flux_centers = (flux_edges_cpu[1:] + flux_edges_cpu[:-1])/2
    flux_pdf_cpu = cupy.asnumpy(pdf_estimator.flux_pdf)
    fisher_cpu = cupy.asnumpy(pdf_estimator.fisher)

    logging_mpi("Reducing to root.", mpi_rank)
    comm.reduce(flux_pdf_cpu, root=0)
    comm.reduce(fisher_cpu, root=0)

    if mpi_rank == 0:
        logging.info("Getting final results")
        _di_idx = np.diag_indices(flux_pdf_cpu.size)
        w = fisher_cpu[_di_idx] == 0
        fisher_cpu[_di_idx][w] = 1

        cov = np.linalg.inv(fisher_cpu)
        flux_pdf = cov@flux_pdf_cpu

        # Save flux pdf fn
        logging.info("Saving to files.")
        fname2save = f"{output_dir}/{output_base}-flux-pdf-estimate.txt"
        zarr_repeated = np.repeat(config_qmle.z_bins, nfbins)
        farr_repeated = np.tile(flux_centers, config_qmle.z_n)
        egauss = np.sqrt(cov.diagonal())

        corr_table = Table([zarr_repeated, farr_repeated, flux_pdf, egauss], \
            names=('z', 'F', 'FPDF', 'e_FPDF'))
        corr_table.write(fname2save, format="ascii", overwrite=True, \
            formats={'z':'%.5e', 'F':'%.5e', 'FPDF':'%.5e', 'e_FPDF':'%.5e'})
        logging.info(f"Flux PDF saved as {fname2save}")

        # Save Fisher
        fname2save = f"{output_dir}/{output_base}-fisher-flux-pdf-estimate.txt"
        np.savetxt(fname2save, fisher_cpu)

        logging.info("DONE!")






















