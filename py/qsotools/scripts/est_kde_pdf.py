import argparse
import logging
from multiprocessing import Pool

import numpy as np
from numba import njit
from astropy.table import Table

import qsotools.io as qio
import qsotools.fiducial as fid
import qsotools.utils as qutil

recip_sqrt_2pi = 1. / np.sqrt(2 * np.pi)


def get_parser():
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ConfigFile", help="Config file")

    parser.add_argument(
        "--nsubsamples", type=int, default=100,
        help="Number of subsamples if input is not Picca.")
    parser.add_argument(
        "--f1", help="First flux bin", type=float, default=-0.5)
    parser.add_argument(
        "--f2", help="Last flux bin", type=float, default=1.75)
    parser.add_argument(
        "--nfbins", help="Number of flux bins", type=float, default=1000)

    parser.add_argument("--min-snr", type=float, default=0, help="Minimum SNR")
    parser.add_argument(
        "--deconvolve", action="store_true",
        help="Deconvolve mean sigma from pdf.")

    # parser.add_argument(
    #     "--sigma1", help="First mean sigma bin", type=float, default=0)
    # parser.add_argument(
    #     "--nsigmabins", help="Number of sigma bins", type=int, default=8)
    # parser.add_argument(
    #     "--dsigma", help="sigma bin size", type=float, default=0.5)

    # parser.add_argument(
    #     "--smooth-noise-sigmaA", type=float, default=20.,
    #     help="Gaussian sigma in A to smooth pipeline noise estimates.")
    parser.add_argument(
        "--no-convert2flux", action="store_true",
        help="Does not convert delta values to flux using FG08.")
    parser.add_argument(
        "--min-nopix", help="Minimum number of pixels in chunk", type=int,
        default=20)

    parser.add_argument("--nproc", type=int, default=None)
    # parser.add_argument(
    #     "--debug", help="Set logger to DEBUG level.", action="store_true")

    return parser


@njit
def get_kde_estimate(x, mus, sigmas):
    result = np.zeros_like(x)
    for (mu, sigma) in zip(mus, sigmas):
        xx = (x - mu) / sigma
        result += np.exp(-xx**2 / 2) * recip_sqrt_2pi / sigma

    return result


class KDE_PDF_Estimator(object):
    def __init__(self, args, config_qmle, flux_centers):
        self.args = args
        self.config_qmle = config_qmle
        self.flux_centers = flux_centers

        self.nfbins = flux_centers.size
        self.z_edges = config_qmle.z_edges

        self.flux_pdf = np.zeros(self.nfbins * self.config_qmle.z_n)
        self.counts = np.zeros(self.nfbins * self.config_qmle.z_n)
        self.kfreq = 2. * np.pi * np.fft.rfftfreq(
            self.nfbins, d=flux_centers[1] - flux_centers[0])

    def deconvolve(self, kde_estim, sigma):
        if not self.args.deconvolve:
            return kde_estim

        kde_estim_k = np.fft.rfft(kde_estim)
        xx = self.kfreq * sigma
        kde_estim_k /= np.exp(-xx**2 / 2)

        return np.fft.irfft(kde_estim_k)

    def getEstimates(self, qso):
        qso.applyMask()
        z_arr = qso.wave / fid.LYA_WAVELENGTH - 1
        z_med = z_arr[qso.size // 2]
        z_bin_no = (
            (z_med - self.config_qmle.z_edges[0]) / self.config_qmle.z_d
        ).astype(int)

        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n - 1:
            logging.debug(f"Skipping z_med={z_med:.2f}")
            return

        mean_snr = np.mean(1. / qso.error)
        if mean_snr < self.args.min_snr:
            return

        if not self.args.no_convert2flux:
            mf = fid.meanFluxFG08(z_arr)
            qso.flux = (1 + qso.flux) * mf
            qso.error *= mf

        i1 = z_bin_no * self.nfbins
        i2 = i1 + self.nfbins
        kde_estim = get_kde_estimate(self.flux_centers, qso.flux, qso.error)
        kde_estim = self.deconvolve(kde_estim, qso.error.mean())
        self.flux_pdf[i1:i2] += kde_estim
        self.counts[i1:i2] += qso.size

    def __call__(self, fname):
        qsos_list = []

        if self.config_qmle.picca_input:
            base, hdus = fname
            f = f"{self.config_qmle.qso_dir}/{base}"
            pfile = qio.PiccaFile(f, 'r', clobber=False)

            for hdu in hdus:
                qsos_list.extend(pfile.readSpectrum(hdu).split(
                    self.config_qmle.z_edges, self.args.min_nopix))

            pfile.close()
        else:
            f = f"{self.config_qmle.qso_dir}/{fname.rstrip()}"
            qsos_list.append(qio.BinaryQSO(f, 'r'))

        for qso in qsos_list:
            self.getEstimates(qso)

        return self.flux_pdf, self.counts


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read Config file
    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir = config_qmle.parameters['OutputDir']
    output_base = (
        f"{config_qmle.parameters['OutputFileBase']}"
        f"-minsnr{args.min_snr:.2f}")

    # Set up logger
    logging.basicConfig()

    # Set up flux
    flux_centers = np.linspace(args.f1, args.f2, args.nfbins)
    flux_subsampler = qutil.SubsampleCov(
        args.nfbins * config_qmle.z_n, 100, is_weighted=True)

    # Read file list file
    fnames_spectra = config_qmle.readFnameSpectra()

    # If files are in Picca format, decompose filename list into
    # Main file & hdus to read in that main file
    if config_qmle.picca_input:
        logging.info("Decomposing filenames to a list of (base, list(hdus)).")
        fnames_spectra = qutil.getPiccaFList(fnames_spectra)

    nfiles = len(fnames_spectra)
    pcounter = qutil.Progress(nfiles)  # Progress tracker
    logging.info(f"There are {nfiles} files.")

    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(
            KDE_PDF_Estimator(args, config_qmle, flux_centers),
            fnames_spectra)
        for f, c in imap_it:
            flux_subsampler.addMeasurement(f, c)
            pcounter.increase()

    mean_fpdf, cov_fdpd = flux_subsampler.getMeanNCov()
    # Save flux pdf fn
    fname2save = f"{output_dir}/{output_base}-flux-pdf-estimate.txt"
    zarr_repeated = np.repeat(config_qmle.z_bins, args.nfbins)
    farr_repeated = np.tile(flux_centers, config_qmle.z_n)
    egauss = np.sqrt(cov_fdpd.diagonal())

    corr_table = Table(
        [zarr_repeated, farr_repeated, mean_fpdf, egauss],
        names=('z', 'F', 'FPDF', 'e_FPDF'))
    corr_table.write(
        fname2save, format="ascii", overwrite=True,
        formats={'z': '%.5e', 'F': '%.5e', 'FPDF': '%.5e', 'e_FPDF': '%.5e'})
    logging.info(f"Flux PDF saved as {fname2save}")

    # # Save Fisher
    # fname2save = f"{output_dir}/{output_base}-jackknife-flux-pdf-cov.txt"
    # np.savetxt(fname2save, cov_fdpd)

    logging.info("DONE!")
