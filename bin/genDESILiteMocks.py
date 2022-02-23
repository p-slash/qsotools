#!/usr/bin/env python
# TODO: 
# Add reading a master file. 
# Load balancing for parallel computing

# This script generates DESI-lite mocks
# Creates mocks in log-space wave grid with 30 km/s size
# Assumes a constant window function with R=3200 which corresponds to ~ 40 km/s
# Adds gaussian noise with sigma=0.25 such that s/n is ~1 per pixel.
from os.path import join as ospath_join
from os      import makedirs as os_makedirs
import time
import argparse
import logging
from multiprocessing import Pool

import numpy as np
import healpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from pkg_resources import resource_filename

import qsotools.mocklib  as lm
import qsotools.specops  as so
from qsotools.io import BinaryQSO, QQFile, PiccaFile
import qsotools.fiducial as fid

PKG_ICDF_Z_TABLE = resource_filename('qsotools', 'tables/invcdf_nz_qso_zmin2.1_zmax4.4.dat')

filename_list = []
RESOMAT = None

def setResolutionMatrix(wave, args, ndiags=11):
    assert args.save_picca
    assert args.fixed_zqso

    Ngrid = wave.size
    Rint = args.specres
    dv = args.pixel_dv
    if args.use_optimal_rmat:
        logging.info("Using optimal resolution matrix.")
        logging.info("Calculating correlation function.")
        z = np.median(wave)/fid.LYA_WAVELENGTH-1
        _, xi = lm.lognPowerSpGH(z, numvpoints=2**16, corr=True)
        xi = xi.ravel()
        xi = np.fft.fftshift(xi)
        logging.info("Calculating optimal rmatrix.")
        return so.getOptimalResolutionMatrix(Ngrid, xi, Rint, dv)
    else:
        logging.info("Using Gaussian resolution matrix.")
        rmat = so.getGaussianResolutionMatrix(wave, Rint)

        if args.oversample_rmat > 1:
            rmat = so.getOversampledRMat(rmat, args.oversample_rmat)
        return rmat

def save_parameters(txt_basefilename, args):
    Parameters_txt = ("Parameters for these mocks\n"
        "Type                 : %s\n"
        "Velocity to Redshift : %s\n"
        "Errors               : %.2f\n"
        "Specres              : %d\n"
        "LowResPixelSize      : %f\n"
        "Seed                 : %d\n"
        "NGrid                : %d\n"
        "GridPixelSize        : %f\n"
        "Redshift Evolution   : %s\n"
        "Catalog used         : %s\n") % ( 
    "Gaussian Mocks" if args.gauss else "Lognormal Mocks", \
    "Logarithmic", \
    # if not args.use_eds_v else "EdS", \
    args.sigma_per_pixel, \
    args.specres, \
    args.pixel_dv, \
    args.seed, \
    args.log2ngrid, \
    args.griddv, \
    "ON" if not args.fixed_zforest else "OFF", \
    args.master_file if args.master_file else "None")

    temp_fname = "%s_parameters.txt" % txt_basefilename
    logging.info(f"Saving parameteres to {temp_fname}")
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()

def save_plots(wch, fch, ech, fnames, args):
    for (w, f, e, fname) in zip(wch, fch, ech, fnames):
        plt.cla()
        plt.plot(w, f, 'b-')
        plt.grid(True, "major")
        plt.plot(w, e, 'r-')
        plt.savefig(ospath_join(args.OutputDir, fname[:-3]+"png"), bbox_inches='tight', dpi=150)

def save_data(wave, fmocks, emocks, fnames, z_qso, dec, ra, args, picca=None):
    if picca:
        for (w, f, e) in zip(wave, fmocks, emocks):
            fname=picca.writeSpectrum(w, f, e, args.specres, z_qso, ra, dec, RESOMAT.T, \
                islinbin=not args.use_logspaced_wave, oversampling=args.oversample_rmat)
            filename_list.append(fname)
    else:
        for (w, f, e, fname) in zip(wave, fmocks, emocks, fnames):
            mfile = BinaryQSO(ospath_join(args.OutputDir, fname), 'w')
            mfile.save(w, f, e, len(w), z_qso, dec, ra, 0., args.specres, args.pixel_dv)

def saveQQFile(ipix, meta1, wave, fluxes, args):
    P = int(ipix/100)
    dir1 = ospath_join(args.OutputDir, f"{P}")
    dir2 = ospath_join(dir1, f"{ipix}")
    os_makedirs(dir1, exist_ok=True)
    os_makedirs(dir2, exist_ok=True)
    fname = ospath_join(dir2, f"lya-transmission-{args.hp_nside}-{ipix}.fits.gz")

    qqfile = QQFile(fname, 'rw')
    qqfile.writeAll(meta1, wave, fluxes)

    return fname

def chunkHelper(i, waves, fluxes, errors, z_qso):
    wave_c, flux_c, err_c = waves[i], fluxes[i], errors[i]

    if args.chunk_dyn:
        wave_c, flux_c, err_c = so.chunkDynamic(wave_c, flux_c, err_c, len(wave_c))
    if args.chunk_fixed:
        NUMBER_OF_CHUNKS = 3
        FIXED_CHUNK_EDGES = np.linspace(fid.LYA_FIRST_WVL, fid.LYA_LAST_WVL, num=NUMBER_OF_CHUNKS+1)
        wave_c, flux_c, err_c = so.divideIntoChunks(wave_c, flux_c, err_c, z_qso[i], FIXED_CHUNK_EDGES)
    else:
        wave_c = [wave_c]
        flux_c = [flux_c]
        err_c  = [err_c]

    return wave_c, flux_c, err_c

# Returns observed wavelength centers
def getDESIwavegrid(args):
    # Set up DESI observed wavelength grid
    if args.use_logspaced_wave:
        logging.info(f"Using logspaced wavelength grid with dv={args.pixel_dv} km/s.")
        base          = np.exp(args.pixel_dv / fid.LIGHT_SPEED)
        npix_desi     = int(np.log(args.desi_w2 / args.desi_w1) / args.pixel_dv * fid.LIGHT_SPEED)+1
        DESI_WAVEGRID = args.desi_w1 * np.power(base, np.arange(npix_desi))
    else:
        logging.info(f"Using linear wavelength grid with dlambda={args.pixel_dlambda} A.")
        npix_desi = int((args.desi_w2 - args.desi_w1) / args.pixel_dlambda) + 1
        DESI_WAVEGRID = args.desi_w1 + np.arange(npix_desi) * args.pixel_dlambda

    DESI_WAVEEDGES = so.createEdgesFromCenters(DESI_WAVEGRID, logspacing=args.use_logspaced_wave)

    return DESI_WAVEGRID, DESI_WAVEEDGES

def _genRNDDec(RNST, N, dec1_deg, dec2_deg):
    asin1 = np.sin(dec1_deg*np.pi/180.)
    asin2 = np.sin(dec2_deg*np.pi/180.)
    rnd_asin = (asin2-asin1)*RNST.random(N)+asin1

    return np.arcsin(rnd_asin) * 180./np.pi

# Returns metadata array and number of pixels
def getMetadata(args):
    # The METADATA HDU contains a binary table with (at least) RA,DEC,Z,TARGETID
    meta_dt = np.dtype([('RA','f8'), ('DEC','f8'), ('Z','f8'),('MOCKID','i8'), \
        ('PIXNUM','i4'), ('COADD_EXPTIME','f8'), ('FLUX_R','f8')])
    dt_list = list(meta_dt.names)
    dt_list.remove('PIXNUM')
    if args.master_file:
        logging.info(f"Reading master file: {args.master_file}")
        master_file = QQFile(args.master_file)
        l1 = master_file.readMetadata()
        master_file.close()

        # Remove low redshift quasars
        zqso_cut_index       = master_file.metadata['Z'] > args.z_quasar_min
        master_file.metadata = master_file.metadata[zqso_cut_index]
        args.nmocks          = master_file.metadata.size

        # Add pixnum field to metadata
        metadata = np.zeros(args.nmocks, dtype=meta_dt)
        for mcol in list(set(l1) & set(dt_list)):
            metadata[mcol] = master_file.metadata[mcol]

        logging.info(f"Number of mocks to generate: {args.nmocks}")
    else:
        logging.info("Generating random metadata.")
        metadata = np.zeros(args.nmocks, dtype=meta_dt)
        metadata['MOCKID'] = np.arange(args.nmocks)
        # Use the same seed for all process to generate the same metadata
        RNST = np.random.default_rng(args.seed)
        # Generate coords in degrees
        metadata['RA']  = RNST.random(args.nmocks) * 360.
        # metadata['DEC'] = (RNST.random(args.nmocks)-0.5) * 180.

        dec1, dec2 = (-20., 80.) if args.desi_dec else (-90., 90.)
        metadata['DEC'] = _genRNDDec(RNST, args.nmocks, dec1, dec2)

        if args.fixed_zqso:
            metadata['Z'] = args.fixed_zqso
        else:
            # Read inverse cumulative distribution function
            # Generate uniform random numbers
            # Use inverse CDF to map these to QSO redshifts
            invcdf, zcdf   = np.genfromtxt(args.invcdf_nz, unpack=True)
            inv_cdf_interp = interp1d(invcdf, zcdf)
            metadata['Z']  = inv_cdf_interp(RNST.uniform(size=args.nmocks))

    logging.info(f"Number of nside for heal pixels: {args.hp_nside}")
    if args.hp_nside:
        npixels = healpy.nside2npix(args.hp_nside)
        # when lonlat=True: RA first, Dec later
        # when lonlat=False (Default): Dec first, RA later
        metadata['PIXNUM'] = healpy.ang2pix(args.hp_nside, \
            metadata['RA'], metadata['DEC'], nest=not args.hp_ring, lonlat=True)
    else:
        npixels = 1
        metadata['PIXNUM'] = 0

    mstrfname = ospath_join(args.OutputDir, "master.fits")
    qqfile = QQFile(mstrfname, 'rw')
    qqfile.writeMetadata(metadata)
    qqfile.close()
    logging.info(f"Saved master metadata to {mstrfname}")

    return metadata, npixels

class MockGenerator(object):
    """docstring for MockGenerator"""
    def __init__(self, args):
        self.args = args
        self.TURNOFF_ZEVO = args.fixed_zforest is not None
        # Set up DESI observed wavelength grid
        self.DESI_WAVEGRID, self.DESI_WAVEEDGES = getDESIwavegrid(args)

    def __call__(self, ipix_meta):
        ipix, meta1 = ipix_meta
        ntemp = meta1['MOCKID'].size
        z_qso = meta1['Z'][:, None]

        if ntemp == 0:
            return []

        # ------------------------------
        # Change the seed with thread no for different randoms across processes
        lya_m = lm.LyaMocks(self.args.seed+ipix, N_CELLS=2**self.args.log2ngrid, DV_KMS=self.args.griddv, \
            GAUSSIAN_MOCKS=self.args.gauss, REDSHIFT_ON=not self.TURNOFF_ZEVO)
        if self.TURNOFF_ZEVO:
            lya_m.setCentralRedshift(self.args.fixed_zforest)
        else:
            lya_m.setCentralRedshift(3.0)

        if self.args.gauss:
            mean_flux_function = fid.meanFluxFG08
        else:
            mean_flux_function = lm.lognMeanFluxGH

        wave, fluxes, errors = lya_m.resampledMocks(ntemp, err_per_final_pixel=self.args.sigma_per_pixel, \
            spectrograph_resolution=self.args.specres, obs_wave_edges=self.DESI_WAVEEDGES, \
            keep_empty_bins=self.args.keep_nolya_pixels)

        # Remove absorption above Lya
        nonlya_ind = wave > fid.LYA_WAVELENGTH * (1+z_qso)
        for i in range(ntemp):
            fluxes[i][nonlya_ind[i]] = 1

        if not self.args.save_full_flux:
            if self.TURNOFF_ZEVO:
                spectrum_z = self.args.fixed_zforest
            else:
                spectrum_z = np.array(wave, dtype=np.double) / fid.LYA_WAVELENGTH - 1
            true_mean_flux = mean_flux_function(spectrum_z)

            fluxes  = fluxes / true_mean_flux - 1
            errors /= true_mean_flux

        # If save-qqfile option is passed, do not save as BinaryQSO files
        # This also means no chunking or removing pixels
        if not self.args.nosave and self.args.save_qqfile:
            fname = saveQQFile(ipix, meta1, wave, fluxes, self.args)

            return [fname]
        
        # Cut Lyman-alpha forest region
        if not self.args.keep_nolya_pixels:
            lya_ind = np.logical_and(wave >= fid.LYA_FIRST_WVL * (1+z_qso), \
                wave <= fid.LYA_LAST_WVL * (1+z_qso))
            forst_bnd = np.logical_and(wave >= fid.LYA_WAVELENGTH*(1+args.z_forest_min), \
                wave <= fid.LYA_WAVELENGTH*(1+args.z_forest_max))
            lya_ind = np.logical_and(lya_ind, forst_bnd)
            waves  = [wave[lya_ind[i]] for i in range(ntemp)]
            fluxes = [fluxes[i][lya_ind[i]] for i in range(ntemp)]
            errors = [errors[i][lya_ind[i]] for i in range(ntemp)]
        else:
            waves = [wave for i in range(ntemp)]

        if self.args.save_picca:
            pcfname = ospath_join(self.args.OutputDir, f"delta-{ipix}.fits.gz")
            pcfile  = PiccaFile(pcfname, 'rw')
        else:
            pcfile = None

        for i in range(ntemp):
            wave_c, flux_c, err_c = chunkHelper(i, waves, fluxes, errors, z_qso)

            nchunks = len(wave_c)
            nid = meta1['MOCKID'][i]

            if not self.args.save_picca:
                fname = ["desilite_seed%d_id%d_%d_z%.1f%s.dat" \
                % (args.seed, nid, nc, z_qso[i], settings_txt) for nc in range(nchunks)]
            else:
                assert nchunks == 1
                if RESOMAT is None:
                    RESOMAT = setResolutionMatrix(wave_c[0], args)
                fname = None

            if not self.args.nosave:
                save_data(wave_c, flux_c, err_c, fname, z_qso[i], meta1['DEC'][i], 
                    meta1['RA'][i], args, pcfile)

            if args.plot:
                save_plots(wave_c, flux_c, err_c, fname, args)

            return fname

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("OutputDir", help="Output directory")
    parser.add_argument("--master-file", help="Master file location. Generate mocks with "\
        "the exact RA, DEC & Z distribution. nmocks option is ignored when this passed.")
    parser.add_argument("--fixed-zqso", help="Generate QSOs at this redshift only.", type=float)
    parser.add_argument("--fixed-zforest", help="Generate forest at this redshift, " \
        "i.e. turns off redshift evolution.", type=float)
    parser.add_argument("--nmocks", help=("Number of mocks to generate. "\
        "Redshift of qso picked at random given n(z). Default: %(default)s"), type=int, default=1)

    parser.add_argument("--save-qqfile", action="store_true", \
        help="Saves in quickquasar fileformat. Spectra are not chunked and all pixels are kept."\
        " Sets sigma-per-pixel=0, specres=0, keep-nolya-pixels=True and save-full-flux=True")
    parser.add_argument("--save-picca", action="store_true", \
        help="Saves in picca fileformat.")
    parser.add_argument("--use-optimal-rmat", action="store_true")
    parser.add_argument("--oversample-rmat", help="Oversampling factor for resolution matrix. "\
        "Pass >1 to get finely space response function.", type=int, default=1)

    parser.add_argument("--seed", help="Seed to generate random numbers. Default: %(default)s", \
        type=int, default=332298)

    parser.add_argument("--sigma-per-pixel", help=("Add Gaussian error to mocks with given sigma. "\
        "Default: %(default)s"), type=float, default=0.7)
    parser.add_argument("--specres", help="Spectral resolution. Default: %(default)s", type=int, \
        default=3200)
    parser.add_argument("--pixel-dv", help=("Pixel size (km/s) of the log-spaced wave grid. "\
        "Default: %(default)s"), type=float, default=30.)
    parser.add_argument("--pixel-dlambda", help=("Pixel size (A) of the linearly-spaced wave grid. "\
        "Default: %(default)s"), type=float, default=0.2)
    parser.add_argument("--use-logspaced-wave", help=("Use log spaced array as final grid. "\
        "Default: %(default)s"), action="store_true")

    parser.add_argument("--desi-w1", help=("Lower wavelength of DESI wave grid in A. "\
        "Default: %(default)s A"), type=float, default=3600.)
    parser.add_argument("--desi-w2", help=("Higher wavelength of DESI wave grid in A. "\
        "Default: %(default)s A"), type=float, default=9800.)
    parser.add_argument("--desi-dec", help="Limit dec to (-20, 80).", action="store_true")
    parser.add_argument("--z-quasar-min", type=float, default=2.1, \
        help="Lowest quasar redshift. Only when created from a catalog. Default: %(default)s")
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=1.9)
    parser.add_argument("--z-forest-max", help="Upper end of the forest. Default: %(default)s", \
        type=float, default=4.3)

    parser.add_argument("--keep-nolya-pixels", action="store_true", \
        help="Instead of removing pixels, set flux=1 for lambda>L_lya")
    parser.add_argument("--invcdf-nz", help="Table for inverse cdf of n(z). Default: %(default)s", \
        default=PKG_ICDF_Z_TABLE)
    
    parser.add_argument("--chunk-dyn",  action="store_true", \
        help="Splits spectrum into three chunks if n>2N/3 or into two chunks if n>N/3.")
    parser.add_argument("--chunk-fixed",  action="store_true", \
        help="Splits spectrum into 3 chunks at fixed rest frame wavelengths")

    parser.add_argument("--nosave", help="Does not save mocks to output when passed", \
        action="store_true")
    parser.add_argument("--plot", help="Saves plots to output when passed", action="store_true")

    parser.add_argument("--gauss", help="Generate Gaussian mocks", action="store_true")
    # parser.add_argument("--without-z-evo", help="Turn off redshift evolution", action="store_true")
    parser.add_argument("--save-full-flux", action="store_true", \
        help="When passed saves flux instead of fluctuations around truth.")

    parser.add_argument("--log2ngrid", help="Number of grid points. Default: %(default)s", \
        type=int, default=18)
    parser.add_argument("--griddv", help="Pixel size of the grid in km/s. Default: %(default)s", \
        type=float, default=2.)

    # healpix support
    parser.add_argument("--hp-nside", type=int, default=0)
    parser.add_argument("--hp-ring", action="store_true", \
        help="Use RING pixel ordering. Default is NESTED.")

    # parallel support
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--debug", help="Set logger to DEBUG level.", action="store_true")
    args = parser.parse_args()
    
    start_time = time.time()

    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)
    RESOMAT = None

    logging.basicConfig(filename=ospath_join(args.OutputDir, f'genthread.log'), \
        level=logging.DEBUG if args.debug else logging.INFO)

    metadata, npixels = getMetadata(args)

    if args.save_qqfile:
        args.sigma_per_pixel = 0
        args.specres = 0
        args.desi_w1 = 3400.0
        args.z_forest_min = 0
        args.keep_nolya_pixels = True
        args.save_full_flux = True

    settings_txt  = '_gaussian' if args.gauss else '_lognormal' 
    # settings_txt += '_noz' if args.without_z_evo else ''

    txt_basefilename  = "%s/desilite_seed%d%s" % (args.OutputDir, args.seed, settings_txt)

    save_parameters(txt_basefilename, args)

    metadata.sort(order='PIXNUM')
    logging.info("Metadata sorted.")

    u_pix, s = np.unique(metadata['PIXNUM'], return_index=True)
    split_meta = np.split(metadata, s[1:])
    logging.info(f"Length of split metadata {len(split_meta)} vs npixels {npixels}.")
    pcounter = Progress(len(split_meta))

    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(MockGenerator(args), zip(u_pix, split_meta))
        for fname in imap_it:
            filename_list.extend(fname)
            pcounter.increase()

    # Save the list of files in a txt
    temp_fname = ospath_join(args.OutputDir, f"file_list_qso.txt") 
    # "%s_filelist.txt" % txt_basefilename
    logging.info(f"Saving chunk spectra file list as {temp_fname}")
    toWrite = open(temp_fname, 'w')
    toWrite.write("%d\n" % len(filename_list))
    for f in filename_list:
        toWrite.write(f +"\n")
    toWrite.close()

    logging.info("DONE!")






