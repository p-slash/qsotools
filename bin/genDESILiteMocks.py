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
import argparse

import numpy as np
import healpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from pkg_resources import resource_filename

import qsotools.mocklib  as lm
import qsotools.specops  as so
from qsotools.io import BinaryQSO, QQFile
import qsotools.fiducial as fid

PKG_ICDF_Z_TABLE = resource_filename('qsotools', 'tables/invcdf_nz_qso_zmin2.1_zmax4.4.dat')

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
        "Redshift Evolution   : %s\n") % ( 
    "Gaussian Mocks" if args.gauss else "Lognormal Mocks", \
    "Logarithmic", \
    # if not args.use_eds_v else "EdS", \
    args.sigma_per_pixel, \
    args.specres, \
    args.pixel_dv, \
    args.seed, \
    args.log2ngrid, \
    args.griddv, \
    "ON") # "ON" if not args.without_z_evo else "OFF")
            
    temp_fname = "%s_parameters.txt" % txt_basefilename
    print("Saving parameteres to", temp_fname)
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

def save_data(wave, fmocks, emocks, fnames, z_qso, dec, ra, args):
    for (w, f, e, fname) in zip(wave, fmocks, emocks, fnames):
        mfile = BinaryQSO(ospath_join(args.OutputDir, fname), 'w')
        mfile.save(w, f, e, len(w), z_qso, dec, ra, 0., args.specres, args.pixel_dv)

def getDESIwavegrid(args):
    # Set up DESI observed wavelength grid
    if args.use_logspaced_wave:
        print(f"Using logspaced wavelength grid with dv={args.pixel_dv} km/s.")
        base            = np.exp(args.pixel_dv / fid.LIGHT_SPEED)
        npix_desi       = int(np.log(args.desi_w2 / args.desi_w1) / args.pixel_dv * fid.LIGHT_SPEED)+1
        DESI_WAVEGRID   = args.desi_w1 * np.power(base, np.arange(npix_desi))
    else:
        print(f"Using linear wavelength grid with dlambda={args.pixel_dlambda} A.")
        npix_desi = int((args.desi_w2 - args.desi_w1) / args.pixel_dlambda) + 1
        DESI_WAVEGRID = args.desi_w1 + np.arange(npix_desi) * args.pixel_dlambda

    return DESI_WAVEGRID

def getMetadata(args):
    # The METADATA HDU contains a binary table with (at least) RA,DEC,Z,MOCKID
    if args.master_file:
        print("Reading master file:", args.master_file, flush=True)
        master_file = QQFile(args.master_file)
        master_file.readMetada()
        master_file.close()

        args.nmocks = master_file.nqso
        metadata = master_file.metadata
        print("Number of mocks to generate:", args.nmocks, flush=True)
    else:
        print("Generating random metadata.", flush=True)
        metadata = np.zeros(args.nmocks, dtype=[('RA', 'f8'), ('DEC', 'f8'), \
            ('Z', 'f8'), ('MOCKID', 'i8'), ('PIXNUM', 'i4')])
        metadata['MOCKID'] = np.arange(args.nmocks)

        # Read inverse cumulative distribution function
        # Generate uniform random numbers
        # Use inverse CDF to map these to QSO redshifts
        invcdf, zcdf    = np.genfromtxt(args.invcdf_nz, unpack=True)
        inv_cdf_interp  = interp1d(invcdf, zcdf)

        # Use the same seed for all process to generate the same metadata
        RNST = np.random.default_rng(args.seed)
        metadata['Z']   = inv_cdf_interp(RNST.uniform(size=args.nmocks))
        metadata['RA']  = RNST.random(args.nmocks) * 2 * np.pi
        metadata['DEC'] = (RNST.random(args.nmocks)-0.5) * np.pi

    print("Number of nside for heal pixels:", args.hp_nside, flush=True)
    if args.hp_nside:
        npixels = healpy.nside2npix(args.hp_nside)
        metadata['PIXNUM'] = healpy.ang2pix(args.hp_nside, -metadata['DEC']+np.pi/2, metadata['RA'])
    else:
        npixels = 1
        metadata['PIXNUM'] = 0

    if args.ithread == 0:
        qqfile = QQFile(ospath_join(args.OutputDir, "master.fits"), 'rw')
        qqfile.writeMetadata(metadata)
        qqfile.close()
        print("Saved master metadata to", ospath_join(args.OutputDir, "master.fits"), flush=True)

    return metadata, npixels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("OutputDir", help="Output directory")
    parser.add_argument("--master-file", help="Master file location. Generate mocks with "\
        "the exact RA, DEC & Z distribution. nmocks option is ignored when this passed.")
    parser.add_argument("--nmocks", help=("Number of mocks to generate. "\
        "Redshift of qso picked at random given n(z). Default: %(default)s"), type=int, default=1)
    parser.add_argument("--save-qqfile", action="store_true", \
        help="Saves in quickquasar fileformat. Spectra are not chunked and all pixels are kept."\
        " Sets sigma-per-pixel=0, specres=0, keep-nolya-pixels=True and save-full-flux=True")
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

    # parallel support
    parser.add_argument("--nthreads", type=int, default=1, \
        help="Must be < # heal pixels. Default: %(default)s")
    parser.add_argument("--ithread", type=int, default=0, \
        help="Must be < nthreads. Default: %(default)s")
    args = parser.parse_args()
    
    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)

    metadata, npixels = getMetadata(args)
    # Set up DESI observed wavelength grid
    DESI_WAVEGRID  = getDESIwavegrid(args)
    DESI_WAVEEDGES = so.createEdgesFromCenters(DESI_WAVEGRID, logspacing=args.use_logspaced_wave)
    
    assert args.ithread < args.nthreads
    assert args.nthreads <= npixels

    if args.save_qqfile:
        args.sigma_per_pixel = 0
        args.specres = 0
        args.keep_nolya_pixels = True
        args.save_full_flux = True

    settings_txt  = '_gaussian' if args.gauss else '_lognormal' 
    # settings_txt += '_noz' if args.without_z_evo else ''

    txt_basefilename  = "%s/desilite_seed%d%s" % (args.OutputDir, args.seed, settings_txt)

    # ------------------------------
    # Iteration
    filename_list = []

    # Change the seed with thread no for different randoms across processes
    lya_m = lm.LyaMocks(args.seed+args.ithread, N_CELLS=2**args.log2ngrid, DV_KMS=args.griddv, \
        GAUSSIAN_MOCKS=args.gauss)
    lya_m.setCentralRedshift(3.0)
    # REDSHIFT_ON=not args.without_z_evo)

    if args.gauss:
        print("Generating Gaussian mocks.", flush=True)
        mean_flux_function = fid.meanFluxFG08
    else:
        print("Generating lognormal mocks.", flush=True)
        mean_flux_function = lm.lognMeanFluxGH

    if args.ithread == 0:
        save_parameters(txt_basefilename, args)
    
    metadata.sort(order='PIXNUM')
    print("Metadata sorted.", flush=True)

    u_pix, s = np.unique(metadata['PIXNUM'], return_index=True)
    split_meta = np.split(metadata, s[1:])
    print(f"Length of split metadata {len(split_meta)} vs npixels {npixels}.", flush=True)

    # parallel support
    dithr = int(len(u_pix)/args.nthreads)
    i1 = dithr * args.ithread
    i2 = len(u_pix) if (args.ithread == args.nthreads-1) else dithr * (1+args.ithread)

    for ui in range(i1, i2):
        ipix = u_pix[ui]
        curr_progress = int(100*(ui-i1)/(i2-i1))
        print_condition = curr_progress%5 == 0

        if print_condition:
            print(f"Working on pixel {ipix}.")
            print(f"Progress: {curr_progress}%", flush=True)

        meta1 = split_meta[ui]
        ntemp = meta1['MOCKID'].size
        z_qso = meta1['Z'][:, None]
        
        if print_condition:
            print(f"Number of qsos in pixel {ipix} is {ntemp}.", flush=True)

        if ntemp == 0:
            continue

        wave, fluxes, errors = lya_m.resampledMocks(ntemp, err_per_final_pixel=args.sigma_per_pixel, \
            spectrograph_resolution=args.specres, obs_wave_edges=DESI_WAVEEDGES, \
            keep_empty_bins=args.keep_nolya_pixels)

        # Remove absorption above Lya
        nonlya_ind = wave > fid.LYA_WAVELENGTH * (1+z_qso)
        for i in range(ntemp):
            fluxes[i][nonlya_ind[i]] = 1

        if not args.save_full_flux:
            # if args.without_z_evo:
            #     spectrum_z = z_qso * np.ones_like(fluxes)
            # else:
            spectrum_z = np.array(wave, dtype=np.double) / fid.LYA_WAVELENGTH - 1
            true_mean_flux = mean_flux_function(spectrum_z)

            fluxes  = fluxes / true_mean_flux - 1
            errors /= true_mean_flux

        # If save-qqfile option is passed, do not save as BinaryQSO files
        # This also means no chunking or removing pixels
        if not args.nosave and args.save_qqfile:
            P = int(ipix/100)
            dir1 = ospath_join(args.OutputDir, f"{P}")
            dir2 = ospath_join(dir1, f"{ipix}")
            os_makedirs(dir1, exist_ok=True)
            os_makedirs(dir2, exist_ok=True)
            fname = ospath_join(dir2, f"lya-transmission-{args.hp_nside}-{ipix}.fits.gz")
            
            if print_condition:
                print(f"Saving file {fname}.", flush=True)

            qqfile = QQFile(fname, 'rw')
            qqfile.writeAll(meta1, wave, fluxes)
            filename_list.extend([fname])
            continue

        # Cut Lyman-alpha forest region
        if not args.keep_nolya_pixels:
            lya_ind = np.logical_and(wave >= fid.LYA_FIRST_WVL * (1+z_qso), \
                wave <= fid.LYA_LAST_WVL * (1+z_qso))
            waves  = [wave[lya_ind[i]] for i in range(ntemp)]
            fluxes = [fluxes[i][lya_ind[i]] for i in range(ntemp)]
            errors = [errors[i][lya_ind[i]] for i in range(ntemp)]
        else:
            waves = [wave for i in range(ntemp)]

        for i in range(ntemp):
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

            nchunks = len(wave_c)
            nid = meta1['MOCKID'][i]
            fname = ["desilite_seed%d_id%d_%d_z%.1f%s.dat" \
                % (args.seed, nid, nc, z_qso[i], settings_txt) for nc in range(nchunks)]

            filename_list.extend(fname)
            if not args.nosave:
                save_data(wave_c, flux_c, err_c, fname, z_qso[i], meta1['DEC'][i], meta1['RA'][i], args)

            if args.plot:
                save_plots(wave_c, flux_c, err_c, fname, args)

    # Save the list of files in a txt
    temp_fname = ospath_join(args.OutputDir, f"file_list_qso-{args.ithread}.txt") # "%s_filelist.txt" % txt_basefilename
    print("Saving chunk spectra file list as ", temp_fname, flush=True)
    toWrite = open(temp_fname, 'w')
    toWrite.write("%d\n" % len(filename_list))
    for f in filename_list:
        toWrite.write(f +"\n")
    toWrite.close()

    print("DONE!")






