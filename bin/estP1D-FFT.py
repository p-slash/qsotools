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
from qsotools.utils import Progress, SubsampleCov

def binPowerSpectra(raw_k, raw_p, k_edges):
    binned_power,  _, binnumber = binned_statistic(raw_k, raw_p, statistic='sum', bins=k_edges)
    counts = np.bincount(binnumber, minlength=len(k_edges)+1)

    return binned_power, counts


def decomposePiccaFname(picca_fname):
    i1 = picca_fname.rfind('[')+1
    i2 = picca_fname.rfind(']')

    basefname = picca_fname[:i1-1]
    hdunum = int(picca_fname[i1:i2])

    return (basefname, hdunum)

def _splitQSO(qso, z_edges, min_nopix):
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

        if tmp_qso.s2n > 0 and wave.size > min_nopix:
            split_qsos.append(tmp_qso)

    return split_qsos

class FFTEstimator(object):
    def __init__(self, args, config_qmle):
        self.args = args
        self.config_qmle = config_qmle

        self.mean_k_skm = np.zeros((self.config_qmle.z_n, self.config_qmle.k_bins.size))
        self.mean_z = np.zeros(self.config_qmle.z_n)
        self.power = np.zeros((self.config_qmle.z_n, self.config_qmle.k_bins.size))
        self.cross_power = np.zeros((self.config_qmle.z_n, self.config_qmle.k_bins.size))
        self.counts = np.zeros_like(self.power)
        self.mean_resolution = np.zeros(self.config_qmle.z_n)
        self.counts_meanreso = np.zeros_like(self.mean_resolution)

        self.min_lengthA_zbin = self.args.skip_ratio *self.config_qmle.z_d * fid.LYA_WAVELENGTH

    def getEstimates(self, qso, qso_2, pad_mult=7):
        this_z_arr = qso.wave/fid.LYA_WAVELENGTH-1
        z_med = this_z_arr[int(qso.size/2)]
        z_bin_no = int((z_med - self.config_qmle.z_edges[0]) / self.config_qmle.z_d)

        assert np.isclose(z_med, self.config_qmle.z_bins[z_bin_no], atol=self.config_qmle.z_d/2)
        if z_bin_no < 0 or z_bin_no > self.config_qmle.z_n-1:
            return

        # assume wavelength is linear
        dlambda = np.diff(qso.wave)
        if not np.allclose(dlambda, dlambda[1]):
            raise Exception("non-equal wavelength grid in qso.")

        dlambda = dlambda[1]
        qso.dv  = fid.LIGHT_SPEED*dlambda/qso.wave[int(qso.size/2)]

        # pad arrays
        qso.flux = np.pad(qso.flux, (0, qso.size*pad_mult))
        # Need to use original size for proper normalization
        # of the power spectrum
        length_in_A = dlambda * qso.size
        # length_in_kms = fid.LIGHT_SPEED*np.log(1+length_in_A/qso.wave[0])

        # Add to mean resolution
        w = qso.reso_kms > 0
        self.mean_resolution[z_bin_no] += np.sum(qso.reso_kms[w])
        self.counts_meanreso[z_bin_no] += np.sum(w)

        # Compute & bin power
        delta_k = np.fft.rfft(qso.flux)
        p1d_f = np.abs(delta_k)**2 * dlambda**2 / length_in_A
        if qso_2 is not None:
            if qso.size != qso_2.size:
                raise Exception("different sized cross delta file")
            qso_2.flux = np.pad(qso_2.flux, (0, qso.size*pad_mult))
            delta_k_2 = np.fft.rfft(qso_2.flux)
            pcross = np.real(delta_k.conj() * delta_k_2)
            pcross *= dlambda**2 / length_in_A
        else:
            pcross = np.zeros_like(p1d_f)

        if self.args.noise_realizations>0:
            pnoise = np.zeros_like(p1d_f)
            for _ in range(self.args.noise_realizations):
                delta_noise = np.pad(np.random.default_rng().normal(0, qso.error), (0, qso.size*pad_mult))
                pnoise += np.abs(np.fft.rfft(delta_noise))**2 * dlambda**2 / length_in_A

            pnoise /= self.args.noise_realizations
            p1d_f -= pnoise

        this_k_arr = 2*np.pi*np.fft.rfftfreq(qso.flux.size, dlambda)
        if self.args.deconv_window:
            # window = getSpectographWindow_k(this_k_arr, qso.specres, qso.dv)**2
            _kx = this_k_arr*dlambda
            window = np.exp(-_kx**2)
            if not self.args.no_tophat:
                window *= np.sinc(_kx/2/np.pi)**2
            p1d_f /= window
            pcross /= window

        # Convert A to km/s
        conversion_A_kms = fid.LYA_WAVELENGTH*(1+z_med) / fid.LIGHT_SPEED # A/(km/s)
        this_k_arr *= conversion_A_kms
        p1d_f /= conversion_A_kms

        # ignore k=0 mode
        jj = 1 #int(qso.flux.size/qso.size)
        p, c = binPowerSpectra(this_k_arr[jj:], p1d_f[jj:], config_qmle.k_edges)

        # if qso_2 is not None:
        pcross /= conversion_A_kms
        pcross, _ = binPowerSpectra(this_k_arr[jj:], pcross[jj:], config_qmle.k_edges)

        # Recalculate error by adding lss to variance
        if self.args.weighted_average:
            var_lss = fid.getLyaFlucErrors(z_med, qso.dv, qso.dv, on_flux=False)
            weight = qso.s2n**2 / (1+var_lss*qso.s2n**2)
        else:
            weight = 1

        self.power[z_bin_no] += p * weight
        self.cross_power[z_bin_no] += pcross * weight
        self.counts[z_bin_no] += c[1:-1] * weight

        self.mean_k_skm[z_bin_no] += binPowerSpectra(this_k_arr[jj:], this_k_arr[jj:], config_qmle.k_edges)[0]* weight
        z_pairs = np.sqrt(np.outer(1+this_z_arr, 1+this_z_arr).ravel())-1
        self.mean_z[z_bin_no] += np.mean(z_pairs) * np.sum(c[1:-1] * weight)

    def _picca_file_call(self, fnames):
        base, hdus = fnames
        f = ospath_join(self.config_qmle.qso_dir, base)
        pfile = qio.PiccaFile(f, 'r')

        if args.indir2:
            base_delta = base.split("/")[-1]
            f2 = ospath_join(self.args.indir2, base_delta)
            pfile_2 = qio.PiccaFile(f2, 'r')

        for hdu in hdus:
            qso = pfile.readSpectrum(hdu)
            dlambda = qso.wave[1]-qso.wave[0]
            min_nopix = self.min_lengthA_zbin/dlambda
            split_qsos = _splitQSO(qso, self.config_qmle.z_edges, min_nopix)

            if args.indir2:
                extname = pfile.fitsfile[hdu].get_extname()

                try:
                    qso_2 = pfile_2.readSpectrum(extname)
                except:
                    logging.error(f"{extname} does not exist in indir2/{base}[{hdu}].")
                    continue

                split_qsos_2 = _splitQSO(qso_2, self.config_qmle.z_edges, min_nopix)
            else:
                split_qsos_2 = [None] * len(split_qsos)

            for qso, qso_2 in zip(split_qsos, split_qsos_2):
                try:
                    self.getEstimates(qso, qso_2)
                except Exception as e:
                    logging.error(f"{e} in {base}[{hdu}]")

        pfile.close()
    def __call__(self, fnames):
        if self.config_qmle.picca_input:
            self._picca_file_call(fnames)
        else:
            for fl in fnames:
                f = ospath_join(self.config_qmle.qso_dir, fl.rstrip())
                bq = qio.BinaryQSO(f, 'r')
                try:
                    self.getEstimates(bq)
                except Exception as e:
                        logging.error(f"{e} in {fl}")

        return self.power, self.cross_power, self.counts, self.mean_resolution, self.counts_meanreso,\
            self.mean_k_skm, self.mean_z


if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ConfigFile", help="Config file")
    parser.add_argument("--indir2",
        help="Cross correlate with the delta files in this directory")
    parser.add_argument("--deconv-window", action="store_true",
        help="Deconvolve window function. Assumes dlambda Gaussian and top-hat.")
    parser.add_argument("--no-tophat", action="store_true",
        help="No tophat correction.")
    # parser.add_argument("--correlation", help="Compute correlation fn instead.")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--skip-ratio", type=float, default=0.3,
        help="Skip ratio for a chunk wrt zbin size")
    parser.add_argument("--noise-realizations", type=int, default=100)
    parser.add_argument("--weighted-average", action="store_true",
        help="Averages power spectrum with mean snr^2")
    parser.add_argument("--nsubsamples", type=int, default=100, \
        help="Number of subsamples if input is not Picca.")
    parser.add_argument("--debug", help="Set logger to DEBUG level.", action="store_true")
    args = parser.parse_args()

    # Set up logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logging.info("Starting")

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    output_dir  = config_qmle.parameters['OutputDir']
    output_base = config_qmle.parameters['OutputFileBase']

    power = np.zeros((config_qmle.z_n, config_qmle.k_bins.size))
    cross_power = np.zeros_like(power)
    counts = np.zeros_like(power)

    mean_resolution = np.zeros(config_qmle.z_n)
    counts_meanreso = np.zeros_like(mean_resolution)

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
    pcounter = Progress(nfiles) # Progress tracker
    logging.info(f"There are {nfiles} files.")

    nsubsamples = nfiles if config_qmle.picca_input else args.nsubsamples
    reso_samples = SubsampleCov(config_qmle.z_n, nsubsamples, is_weighted=True)
    p1d_samples = SubsampleCov(config_qmle.z_n*config_qmle.k_bins.size, nsubsamples, is_weighted=True)
    cross_samples = SubsampleCov(config_qmle.z_n*config_qmle.k_bins.size, nsubsamples, is_weighted=True)
    mean_k_skm = SubsampleCov(config_qmle.z_n*config_qmle.k_bins.size, nsubsamples, is_weighted=True)
    mean_z = SubsampleCov(config_qmle.z_n, nsubsamples, is_weighted=True)

    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(FFTEstimator(args, config_qmle), fnames_spectra)

        for p1, pc1, c1, mean_res, counts_mreso, k1, z1 in imap_it:
            reso_samples.addMeasurement(mean_res, counts_mreso)
            p1d_samples.addMeasurement(p1.ravel(), c1.ravel())
            cross_samples.addMeasurement(pc1.ravel(), c1.ravel())

            mean_k_skm.addMeasurement(k1.ravel(), c1.ravel())
            mean_z.addMeasurement(z1, c1.sum(axis=1))
            
            pcounter.increase()

    # Loop is done. Now average results
    mean_p1d, cov_p1d = p1d_samples.getMeanNCov()
    mean_cross, cov_cross = cross_samples.getMeanNCov()
    mean_reso, cov_reso = reso_samples.getMeanNCov()

    mean_k_skm = mean_k_skm.getMean()
    mean_z = mean_z.getMean()

    # Mean resolution
    err_reso = np.sqrt(cov_reso.diagonal())
    meanres_filename = ospath_join(output_dir, output_base+"-mean-resolution.txt")
    meanres_table = Table([config_qmle.z_bins, mean_reso, err_reso], names=('z', 'R', 'e_R'))
    meanres_table.write(meanres_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'R':'%.1f', 'e_R':'%.1f'}, overwrite=True)
    logging.info(f"Mean R saved as {meanres_filename}")

    # Save power spectrum
    err_p1d = np.sqrt(cov_p1d.diagonal())
    err_c1d = np.sqrt(cov_cross.diagonal())
    p1d_filename = ospath_join(output_dir, output_base+"-p1d-fft-estimate.txt")

    zarr_repeated = np.repeat(mean_z, config_qmle.k_bins.size)
    karr_repeated = mean_k_skm # np.tile(config_qmle.k_bins, config_qmle.z_n)

    power_table = Table(
        [zarr_repeated, karr_repeated,
        mean_p1d, err_p1d,
        mean_cross, err_c1d],
        names=('z', 'k', 'P1D', 'e_p1d', 'Cross', 'e_cross'))
    power_table.write(p1d_filename, format='ascii.fixed_width', \
        formats={'z':'%.1f', 'k':'%.5e', 'P1D':'%.5e', 'Cross':'%.5e', 'e_p1d':'%.5e', 'e_cross':'%.5e'},
        overwrite=True)
    print("P1D saved as ", p1d_filename)

    # Save covariance
    cov_filename = ospath_join(output_dir, output_base+"-cov-p1d-fft-estimate.txt")
    np.savetxt(cov_filename, cov_p1d)

    if args.indir2:
        cov_filename = ospath_join(output_dir, output_base+"-cov-cross-fft-estimate.txt")
        np.savetxt(cov_filename, cov_cross)




