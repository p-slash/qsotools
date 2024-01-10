from os.path import join as ospath_join
from os import makedirs as os_makedirs
import argparse
import logging
from multiprocessing import Pool

from numba import njit
import numpy as np
from healpy import nside2npix, ang2pix

from pkg_resources import resource_filename

import qsotools.mocklib as lm
import qsotools.specops as so
import qsotools.fiducial as fid
from qsotools.io import BinaryQSO, QQFile, PiccaFile
from qsotools.utils import Progress

PKG_ICDF_Z_TABLE = resource_filename(
    'qsotools', 'tables/invcdf_nz_qso_zmin2.1_zmax4.4.dat')


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("OutputDir", help="Output directory")
    parser.add_argument(
        "--master-file",
        help=("Master file location. Generate mocks with the exact RA, DEC & Z"
              " distribution. nmocks option is ignored when this is passed."))
    parser.add_argument(
        "--fixed-zforest", type=float,
        help="Generate forest at this redshift & turn off redshift evolution.")
    parser.add_argument(
        "--nmocks", type=int, default=1,
        help="Number of mocks to generate.")

    parser.add_argument(
        "--save-qqfile", action="store_true",
        help=("Saves in quickquasar fileformat. Spectra are not chunked and "
              "all pixels are kept. Sets sigma-per-pixel=0, specres=0, "
              "keep-nolya-pixels=True and save-full-flux=True"))
    parser.add_argument(
        "--save-picca", action="store_true", help="Saves in picca fileformat.")
    parser.add_argument("--use-optimal-rmat", action="store_true")
    parser.add_argument(
        "--oversample-rmat", type=int, default=1,
        help=("Oversampling factor for resolution matrix. "
              "Pass >1 to get finely space response function."))

    parser.add_argument(
        "--seed", type=int, default=332298,
        help="Seed to generate random numbers.")

    parser.add_argument(
        "--sigma-per-pixel", type=float, default=0.7,
        help="Add Gaussian error to mocks with given sigma.")
    parser.add_argument(
        "--specres", type=int, default=3200, help="Spectral resolution.")
    parser.add_argument(
        "--pixel-dv", type=float, default=30.,
        help="Pixel size (km/s) of the log-spaced wave grid.")
    parser.add_argument(
        "--pixel-dlambda", type=float, default=0.2,
        help="Pixel size (A) of the linearly-spaced wave grid.")
    parser.add_argument(
        "--use-logspaced-wave", action="store_true",
        help="Use log spaced array as final grid.")

    parser.add_argument(
        "--desi-w1", type=float, default=3500.,
        help="Lower wavelength of DESI wave grid in A.")
    parser.add_argument(
        "--desi-w2", type=float, default=10000.,
        help="Higher wavelength of DESI wave grid in A.")
    parser.add_argument(
        "--desi-dec", action="store_true", help="Limit dec to (-25, 85).")

    parser.add_argument(
        "--fixed-zqso", type=float,
        help="Generate QSOs at this redshift only. Overrides all.")
    parser.add_argument(
        "--use-analytic-cdf", action="store_true",
        help="Uses an analytic CDF for quasar redshifts.")
    parser.add_argument(
        "--z-quasar-min", type=float, default=2.1,
        help="Lowest quasar redshift.")
    parser.add_argument(
        "--z-quasar-max", type=float, default=6.0,
        help="Maximum quasar redshift.")
    parser.add_argument(
        "--z-forest-min", type=float, default=1.9,
        help="Lower end of the forest.")
    parser.add_argument(
        "--z-forest-max", type=float, default=5.3,
        help="Upper end of the forest.")

    parser.add_argument(
        "--keep-nolya-pixels", action="store_true",
        help="Instead of removing pixels, set flux=1 for lambda>L_lya")
    parser.add_argument(
        "--invcdf-nz", default=PKG_ICDF_Z_TABLE,
        help="Table for inverse cdf of n(z).")

    parser.add_argument(
        "--chunk-dyn", action="store_true",
        help=("Splits spectrum into three chunks "
              "if n>2N/3 or into two chunks if n>N/3."))
    parser.add_argument(
        "--chunk-fixed", action="store_true",
        help="Splits spectrum into 3 chunks at fixed rest frame wavelengths")

    parser.add_argument(
        "--nosave", help="Does not save mocks to output when passed",
        action="store_true")

    parser.add_argument(
        "--gauss", help="Generate Gaussian mocks", action="store_true")
    parser.add_argument(
        "--save-full-flux", action="store_true",
        help="When passed saves flux instead of fluctuations around truth.")

    parser.add_argument(
        "--log2ngrid", help="Number of grid points.",
        type=int, default=18)
    parser.add_argument(
        "--griddv", help="Pixel size of the grid in km/s.",
        type=float, default=2.)

    # healpix support
    parser.add_argument("--hp-nside", help="NSIDE", type=int, default=16)
    parser.add_argument(
        "--hp-ring", action="store_true",
        help="Use RING pixel ordering. Default is NESTED.")

    # parallel support
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument(
        "--debug", help="Set logger to DEBUG level.", action="store_true")

    return parser


def save_parameters(txt_basefilename, args):
    mock_type = "Gaussian Mocks" if args.gauss else "Lognormal Mocks"
    z_evo_txt = "ON" if not args.fixed_zforest else "OFF"
    catalog_name = args.master_file if args.master_file else "None"
    Parameters_txt = (
        "Parameters for these mocks\n"
        f"Type                 : {mock_type}\n"
        f"Velocity to Redshift : Logarithmic\n"
        f"Errors               : {args.sigma_per_pixel:.2f}\n"
        f"Specres              : {args.specres}\n"
        f"LowResPixelSize      : {args.pixel_dv}\n"
        f"Seed                 : {args.seed}\n"
        f"log2NGrid            : {args.log2ngrid}\n"
        f"GridPixelSize        : {args.griddv}\n"
        f"Redshift Evolution   : {z_evo_txt}\n"
        f"Catalog used         : {catalog_name}\n"
    )

    temp_fname = "%s_parameters.txt" % txt_basefilename
    logging.info(f"Saving parameteres to {temp_fname}")
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()


# Returns observed wavelength centers
def getDESIwavegrid(args):
    # Set up DESI observed wavelength grid
    if args.use_logspaced_wave:
        logging.info(
            f"Using logspaced wavelength grid, dv={args.pixel_dv} km/s.")
        base = np.exp(args.pixel_dv / fid.LIGHT_SPEED)
        npix_desi = int(np.log(
            args.desi_w2 / args.desi_w1
        ) * fid.LIGHT_SPEED / args.pixel_dv)
        DESI_WAVEGRID = args.desi_w1 * np.power(base, np.arange(npix_desi))
    else:
        logging.info(
            f"Using linear wavelength grid, dlambda={args.pixel_dlambda} A.")
        npix_desi = int((args.desi_w2 - args.desi_w1) / args.pixel_dlambda)
        DESI_WAVEGRID = (
            np.arange(npix_desi) * args.pixel_dlambda + args.desi_w1)

    DESI_WAVEEDGES = so.createEdgesFromCenters(
        DESI_WAVEGRID, logspacing=args.use_logspaced_wave)

    return DESI_WAVEGRID, DESI_WAVEEDGES


def _genRNDDec(RNST, N, dec1_deg, dec2_deg):
    asin1 = np.sin(dec1_deg * np.pi / 180.)
    asin2 = np.sin(dec2_deg * np.pi / 180.)
    rnd_asin = (asin2 - asin1) * RNST.random(N) + asin1

    return np.arcsin(rnd_asin) * 180. / np.pi


# Returns metadata array and number of pixels
def getMetadata(args, rng):
    # The METADATA HDU contains a binary table
    # with (at least) RA, DEC, Z, TARGETID

    if args.master_file:
        logging.info(f"Reading master file: {args.master_file}")
        master_file = QQFile(args.master_file, clobber=False)
        master_file.readMetadata()
        master_file.close()

        # Remove low redshift quasars
        zqso_cut_index = master_file.metadata['Z'] > args.z_quasar_min
        zqso_cut_index &= master_file.metadata['Z'] < args.z_quasar_max
        master_file.metadata = master_file.metadata[zqso_cut_index]
        args.nmocks = master_file.metadata.size

        # Add pixnum field to metadata
        metadata = master_file.metadata

        logging.info(f"Number of mocks to generate: {args.nmocks}")
    else:
        logging.info("Generating random metadata.")
        metadata = np.zeros(args.nmocks, dtype=QQFile.meta_dt)
        metadata['MOCKID'] = np.arange(args.nmocks)
        # Generate coords in degrees
        metadata['RA'] = rng.random(args.nmocks) * 360.
        # metadata['DEC'] = (rng.random(args.nmocks)-0.5) * 180.

        dec1, dec2 = (-25., 85.) if args.desi_dec else (-90., 90.)
        metadata['DEC'] = _genRNDDec(rng, args.nmocks, dec1, dec2)

        if args.fixed_zqso:
            metadata['Z'] = args.fixed_zqso
        else:
            # Read inverse cumulative distribution function
            # Generate uniform random numbers
            # Use inverse CDF to map these to QSO redshifts

            GenZ = lm.RedshiftGenerator(
                args.invcdf_nz, args.z_quasar_min, args.z_quasar_max,
                args.use_analytic_cdf)
            metadata['Z'] = GenZ.generate(rng, args.nmocks)

    logging.info(f"Number of nside for heal pixels: {args.hp_nside}")
    if args.hp_nside:
        npixels = nside2npix(args.hp_nside)
        # when lonlat=True: RA first, Dec later
        # when lonlat=False (Default): Dec first, RA later
        metadata['PIXNUM'] = ang2pix(
            args.hp_nside, metadata['RA'], metadata['DEC'],
            nest=not args.hp_ring, lonlat=True)
    else:
        npixels = 1
        metadata['PIXNUM'] = 0

    header = {'HPXNSIDE': args.hp_nside, 'HPXNEST': not args.hp_ring}
    mstrfname = ospath_join(args.OutputDir, "master.fits")
    qqfile = QQFile(mstrfname, 'rw')
    qqfile.writeMetadata(metadata, header)
    qqfile.close()
    logging.info(f"Saved master metadata to {mstrfname}")

    return metadata, npixels


def setResolutionMatrix(wave, args, ndiags=11):
    assert args.save_picca
    assert args.fixed_zqso

    Ngrid = wave.size
    Rint = args.specres
    dv = args.pixel_dv
    if args.use_optimal_rmat:
        logging.info("Using optimal resolution matrix.")
        logging.info("Calculating correlation function.")
        z = np.median(wave) / fid.LYA_WAVELENGTH - 1
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


def chunkHelper(i, waves, fluxes, errors, z_qso, args):
    wave_c, flux_c, err_c = waves[i], fluxes[i], errors[i]

    if args.chunk_dyn:
        wave_c, flux_c, err_c = so.chunkDynamic(
            wave_c, flux_c, err_c, len(wave_c))
    if args.chunk_fixed:
        NUMBER_OF_CHUNKS = 3
        FIXED_CHUNK_EDGES = np.linspace(
            fid.LYA_FIRST_WVL, fid.LYA_LAST_WVL, num=NUMBER_OF_CHUNKS + 1)
        wave_c, flux_c, err_c = so.divideIntoChunks(
            wave_c, flux_c, err_c, z_qso[i], FIXED_CHUNK_EDGES)
    else:
        wave_c = [wave_c]
        flux_c = [flux_c]
        err_c = [err_c]

    return wave_c, flux_c, err_c


def save_data(
        wave, fmocks, emocks, fnames, z_qso, dec, ra, args, rmat, picca, tid
):
    if picca:
        fnames = []
        for (w, f, e) in zip(wave, fmocks, emocks):
            fname = picca.writeSpectrum(
                tid, w, f, e, args.specres, z_qso, ra, dec, rmat.T,
                islinbin=not args.use_logspaced_wave,
                oversampling=args.oversample_rmat)
            fnames.append(fname)
    else:
        for (w, f, e, fname) in zip(wave, fmocks, emocks, fnames):
            mfile = BinaryQSO(ospath_join(args.OutputDir, fname), 'w')
            mfile.save(
                w, f, e, len(w), z_qso, dec, ra, 0.,
                args.specres, args.pixel_dv)

    return fnames


def saveQQFile(ipix, meta1, wave, fluxes, args, data_dla=None):
    P = ipix // 100
    dir1 = ospath_join(args.OutputDir, f"{P}")
    dir2 = ospath_join(dir1, f"{ipix}")
    os_makedirs(dir1, exist_ok=True)
    os_makedirs(dir2, exist_ok=True)
    fname = ospath_join(
        dir2, f"lya-transmission-{args.hp_nside}-{ipix}.fits.gz")

    if args.nosave:
        return fname

    qqfile = QQFile(fname, 'rw')
    header = {'HPXNSIDE': args.hp_nside, 'HPXNEST': not args.hp_ring}
    qqfile.writeAll(meta1, header, wave, fluxes, data_dla)

    return fname


@njit
def remove_above_lya_absorption(wave, fluxes, z_qso):
    nmocks = fluxes.shape[0]
    # Remove absorption above Lya
    nonlya_ind = wave > (fid.LYA_WAVELENGTH * (1 + z_qso))
    for i in range(nmocks):
        fluxes[i][nonlya_ind[i]] = 1

    return fluxes


class MockGenerator(object):
    def __init__(self, args, dla_sampler):
        self.args = args
        self.dla_sampler = dla_sampler
        self.TURNOFF_ZEVO = args.fixed_zforest is not None
        # Set up DESI observed wavelength grid
        self.DESI_WAVEGRID, self.DESI_WAVEEDGES = getDESIwavegrid(args)

        self.sett_txt = '_gaussian' if self.args.gauss else '_lognormal'
        if self.args.gauss:
            self.mean_flux_function = fid.meanFluxFG08
        else:
            self.mean_flux_function = lm.lognMeanFluxGH

    def generate_wfe_hpx(self, seed, mockids, n_one_iter=100):
        nmocks = mockids.size
        lya_m = lm.LyaMocks(
            seed,
            N_CELLS=2**self.args.log2ngrid,
            DV_KMS=self.args.griddv,
            GAUSSIAN_MOCKS=self.args.gauss,
            REDSHIFT_ON=not self.TURNOFF_ZEVO,
            dla_sampler=self.dla_sampler
        )

        if self.TURNOFF_ZEVO:
            lya_m.setCentralRedshift(self.args.fixed_zforest)
        else:
            lya_m.setCentralRedshift(3.0)

        nwave = self.DESI_WAVEGRID.size
        n_iter = int(nmocks / n_one_iter) + 1
        n_gen_mocks = 0

        wave = self.DESI_WAVEGRID.copy()
        fluxes = np.empty((nmocks, nwave))
        if self.args.save_qqfile:
            errors = None
        else:
            errors = np.empty_like(fluxes)

        data_dlas = []

        for _ in range(n_iter):
            rem_mocks = nmocks - n_gen_mocks

            if rem_mocks <= 0:
                break

            nthis_mock = min(n_one_iter, rem_mocks)
            _slice = np.s_[n_gen_mocks:n_gen_mocks + nthis_mock]
            n_gen_mocks += nthis_mock

            _, _f, _e = lya_m.resampledMocks(
                nthis_mock,
                err_per_final_pixel=self.args.sigma_per_pixel,
                spectrograph_resolution=self.args.specres,
                obs_wave_edges=self.DESI_WAVEEDGES,
                keep_empty_bins=self.args.keep_nolya_pixels,
                mockids=mockids[_slice]
            )

            fluxes[_slice] = _f
            data_dlas.append(lya_m.data_dlas)

            if self.args.save_qqfile:
                continue

            errors[_slice] = _e

        if self.dla_sampler:
            data_dlas = np.concatenate(data_dlas)
        else:
            data_dlas = None

        return wave, fluxes, errors, data_dlas

    def divide_by_mean_flux(self, wave, fluxes, errors):
        if self.args.save_full_flux:
            return fluxes, errors

        if self.TURNOFF_ZEVO:
            spectrum_z = self.args.fixed_zforest
        else:
            spectrum_z = wave / fid.LYA_WAVELENGTH - 1

        true_mean_flux = self.mean_flux_function(spectrum_z)

        fluxes = fluxes / true_mean_flux - 1
        errors /= true_mean_flux

        return fluxes, errors

    def cut_lya_forest_region(self, wave, fluxes, errors, z_qso):
        nmocks = fluxes.shape[0]
        if not self.args.keep_nolya_pixels:
            lya_ind = np.logical_and(
                wave >= fid.LYA_FIRST_WVL * (1 + z_qso),
                wave <= fid.LYA_LAST_WVL * (1 + z_qso))
            forst_bnd = np.logical_and(
                wave >= fid.LYA_WAVELENGTH * (1 + self.args.z_forest_min),
                wave <= fid.LYA_WAVELENGTH * (1 + self.args.z_forest_max))
            lya_ind = np.logical_and(lya_ind, forst_bnd)

            waves = [wave[lya_ind[i]] for i in range(nmocks)]
            fluxes = [fluxes[i][lya_ind[i]] for i in range(nmocks)]
            errors = [errors[i][lya_ind[i]] for i in range(nmocks)]
        else:
            waves = [wave for i in range(nmocks)]

        return waves, fluxes, errors

    def save_nonqq_files(
            self, imock, waves, fluxes, errors, z_qso, meta1, pcfile
    ):
        wave_c, flux_c, err_c = chunkHelper(
            imock, waves, fluxes, errors, z_qso, self.args)

        nchunks = len(wave_c)
        nid = meta1['MOCKID'][imock]
        z_qso_i = z_qso[imock, 0]
        dec, ra = meta1['DEC'][imock], meta1['RA'][imock]
        mockid = meta1['MOCKID'][imock]

        if not self.args.save_picca:
            def _get_bq_fname(nc):
                return (f"desilite_seed{self.args.seed}_id{nid}_{nc}"
                        f"_z{z_qso_i:.1f}{self.sett_txt}.dat")

            fname = [_get_bq_fname(nc) for nc in range(nchunks)]
            RESOMAT = None
        else:
            assert nchunks == 1
            RESOMAT = setResolutionMatrix(wave_c[0], self.args)
            fname = None

        if not self.args.nosave:
            fname = save_data(
                wave_c, flux_c, err_c, fname, z_qso_i, dec, ra,
                self.args, RESOMAT, pcfile, mockid)

        return fname

    def trim_dlas(self, data_dlas, mockids, z_qso):
        if data_dlas is None:
            return None

        new_data_dlas = []

        for jj, targetid in enumerate(mockids):
            w = data_dlas['MOCKID'] == targetid
            this_dlas = data_dlas[w]
            zdlas = this_dlas['Z_DLA_NO_RSD']
            w2 = (
                fid.LYA_WAVELENGTH * (1 + zdlas) > 910. * (1 + z_qso[jj])
            ) & (zdlas < z_qso[jj])
            new_data_dlas.append(this_dlas[w2])

        return np.concatenate(new_data_dlas)

    def __call__(self, ipix_meta):
        ipix, seed, meta1 = ipix_meta
        nmocks = meta1['MOCKID'].size
        z_qso = meta1['Z'][:, None]

        if nmocks == 0:
            return [], nmocks

        wave, fluxes, errors, data_dlas = self.generate_wfe_hpx(
            seed, meta1['MOCKID'])

        data_dlas = self.trim_dlas(data_dlas, meta1['MOCKID'], meta1['Z'])

        # Remove absorption above Lya
        fluxes = remove_above_lya_absorption(wave, fluxes, z_qso)

        # Divide by mean flux if required
        fluxes, errors = self.divide_by_mean_flux(wave, fluxes, errors)

        # If save-qqfile option is passed, do not save as BinaryQSO files
        # This also means no chunking or removing pixels
        if self.args.save_qqfile:
            fname = saveQQFile(ipix, meta1, wave, fluxes, self.args, data_dlas)

            return [fname], nmocks, data_dlas

        # Cut Lyman-alpha forest region and convert wave to waves = [wave]
        waves, fluxes, errors = self.cut_lya_forest_region(
            wave, fluxes, errors, z_qso)

        if self.args.save_picca:
            pcfname = ospath_join(self.args.OutputDir, f"delta-{ipix}.fits.gz")
            pcfile = PiccaFile(pcfname, 'rw')
        else:
            pcfile = None

        all_fnames = []

        for i in range(nmocks):
            fname = self.save_nonqq_files(
                i, waves, fluxes, errors, z_qso, meta1, pcfile)

            if fname:
                all_fnames.extend(fname)

        return all_fnames, nmocks, data_dlas


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG if args.debug else logging.INFO)

    rng = np.random.default_rng(args.seed)
    metadata, npixels = getMetadata(args, rng)

    if args.save_qqfile:
        args.sigma_per_pixel = 0
        args.specres = 0
        args.desi_w1 = 3500.0
        args.z_forest_min = 0
        args.keep_nolya_pixels = True
        args.save_full_flux = True
        dla_sampler = lm.DLASampler()
    else:
        dla_sampler = None

    sett_txt = '_gaussian' if args.gauss else '_lognormal'
    # sett_txt += '_noz' if args.without_z_evo else ''

    txt_basefilename = f"{args.OutputDir}/desilite_seed{args.seed}{sett_txt}"

    save_parameters(txt_basefilename, args)

    metadata.sort(order='PIXNUM')
    logging.info("Metadata sorted.")

    u_pix, s = np.unique(metadata['PIXNUM'], return_index=True)
    split_meta = np.split(metadata, s[1:])
    logging.info(
        f"Length of split metadata {len(split_meta)} vs npixels {npixels}.")
    pcounter = Progress(metadata.size)

    seeds = rng.choice(2**32, size=u_pix.size, replace=False)

    filename_list = []
    master_dla_catalog = []
    with Pool(processes=args.nproc) as pool:
        imap_it = pool.imap(
            MockGenerator(args, dla_sampler),
            zip(u_pix, seeds, split_meta)
        )
        for fname, nmocks, data_dla in imap_it:
            filename_list.extend(fname)
            master_dla_catalog.append(data_dla)
            pcounter.increase(nmocks)

    if master_dla_catalog[0] is not None:
        mdla_fts = QQFile(
            ospath_join(args.OutputDir, "master_dla_catalog.fits"), "rw")
        logging.info(f"Saving master DLA catalog as {mdla_fts.fname}")
        master_dla_catalog = np.concatenate(master_dla_catalog)
        mdla_fts.writeDLAExtention(master_dla_catalog)
        mdla_fts.close()

    # Save the list of files in a txt
    temp_fname = ospath_join(args.OutputDir, "file_list_qso.txt")
    # "%s_filelist.txt" % txt_basefilename
    logging.info(f"Saving chunk spectra file list as {temp_fname}")
    toWrite = open(temp_fname, 'w')
    toWrite.write("%d\n" % len(filename_list))
    for f in filename_list:
        toWrite.write(f + "\n")
    toWrite.close()

    logging.info("DONE!")
