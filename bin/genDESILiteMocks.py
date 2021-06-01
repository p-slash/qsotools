#!/usr/bin/env python
# TODO: 
# Add reading a master file. 
# finish BinaryQSO support


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
    "Logarithmic" if not args.use_eds_v else "EdS", \
    args.sigma_per_pixel, \
    args.specres, \
    args.pixel_dv, \
    args.seed, \
    args.ngrid, \
    args.griddv, \
    "ON" if not args.without_z_evo else "OFF")
            
    temp_fname = "%s_parameters.txt" % txt_basefilename
    print("Saving parameteres to", temp_fname)
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()

def save_plots(wch, fch, ech, fnames, args):
    for (f, e, fname) in zip(fch, ech, fnames):
        plt.cla()
        plt.plot(wch, f, 'b-')
        plt.grid(True, "major")
        plt.plot(wch, e, 'r-')
        plt.savefig(ospath_join(args.Outputdir, fname[:-3]+"png"), bbox_inches='tight', dpi=300)

def save_data(wave, fmocks, emocks, fnames, z_qso, dec, ra, args):
    for (w, f, e, fname) in zip(wave, fmocks, emocks, fnames):
        mfile = BinaryQSO(ospath_join(args.Outputdir, fname), 'w')
        mfile.save(w, f, e, len(w), z_qso, dec, ra, 0., args.specres, args.pixel_dv)

def getDESIwavegrid(args):
    # Set up DESI observed wavelength grid
    if args.use_logspaced_wave:
        base            = np.exp(args.pixel_dv / fid.LIGHT_SPEED)
        npix_desi       = int(np.log(args.desi_w2 / args.desi_w1) / args.pixel_dv * fid.LIGHT_SPEED)+1
        DESI_WAVEGRID   = args.desi_w1 * np.power(base, np.arange(npix_desi))
    else:
        npix_desi = int((args.desi_w2 - args.desi_w1) / args.pixel_dlambda) + 1
        DESI_WAVEGRID = args.desi_w1 + np.arange(npix_desi) * args.pixel_dlambda

    return DESI_WAVEGRID

def getMetadata(args, RNST):
    # The METADATA HDU contains a binary table with (at least) RA,DEC,Z,MOCKID
    metadata = np.zeros(args.nmocks, dtype=[('RA', 'f8'), ('DEC', 'f8'), \
        ('Z', 'f8'), ('MOCKID', 'i8'), ('IPIX', 'i4')])
    metadata['MOCKID'] = np.arange(args.nmocks)

    # Read inverse cumulative distribution function
    # Generate uniform random numbers
    # Use inverse CDF to map these to QSO redshifts
    invcdf, zcdf    = np.genfromtxt(args.invcdf_nz, unpack=True)
    inv_cdf_interp  = interp1d(invcdf, zcdf)
    metadata['Z']   = inv_cdf_interp(RNST.uniform(size=args.nmocks))
    metadata['RA']  = RNST.random(args.nmocks) * 2 * np.pi
    metadata['DEC'] = (RNST.random(args.nmocks)-0.5) * np.pi

    if args.hp_nside:
        npixels = healpy.nside2npix(args.nside)
        metadata['IPIX'] = healpy.ang2pix(args.hp_nside, -metadata['DEC']+np.pi/2, metadata['RA'])
    else:
        npixels = 1

    return metadata, npixels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("Outputdir", help="Output directory")
    parser.add_argument("--nmocks", help=("Number of mocks to generate. "\
        "Redshift of qso picked at random given n(z). Default: %(default)s"), type=int, default=1)
    parser.add_argument("--seed", help="Seed to generate random numbers. Default: %(default)s", \
        type=int, default=332298)
        
    parser.add_argument("--sigma-per-pixel", help=("Add Gaussian error to mocks with given sigma. "\
        "Default: %(default)s"), type=float, default=0.7)
    parser.add_argument("--specres", help="Spectral resolution. Default: %(default)s", type=int, \
        default=3200)
    parser.add_argument("--pixel-dv", help=("Pixel size (km/s) of the log-spaced wave grid. "\
        "Default: %(default)s"), type=float, default=30.)
    parser.add_argument("--pixel-dlambda", help=("Pixel size (A) of the linearly-spaced wave grid. "\
        "Default: %(default)s"), type=float, default=0.5)
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
    parser.add_argument("--without-z-evo", help="Turn off redshift evolution", action="store_true")
    parser.add_argument("--save-full-flux", action="store_true", \
        help="When passed saves flux instead of fluctuations around truth.")

    parser.add_argument("--use-eds-v", \
        help="Use EdS wavelength grid. Default is False (i.e. Logarithmic spacing).", \
        action="store_true")
    parser.add_argument("--ngrid", help="Number of grid points. Default is 2^16", type=int, \
        default=2**18)
    parser.add_argument("--griddv", help="Pixel size of the grid in km/s. Default: %(default)s", \
        type=float, default=2.)

    # healpix support
    parser.add_argument("--hp-nside", type=int, default=0)
    # parallel support
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--ithread", type=int, default=0)
    parser.add_argument("--save-qqfile", action="store_true")
    args = parser.parse_args()

    # Change the seed with thread no for different randoms across processes
    RNST = np.random.default_rng(args.seed + args.ithread)
    metadata, npixels = getMetadata(args, RNST)
    
    assert args.ithread < args.nthreads
    assert args.nthreads <= npixels

    # Create/Check directory
    os_makedirs(args.Outputdir, exist_ok=True)
    
    settings_txt  = '_gaussian' if args.gauss else '_lognormal' 
    settings_txt += '_noz' if args.without_z_evo else ''

    txt_basefilename  = "%s/desilite_seed%d%s" % (args.Outputdir, args.seed, settings_txt)

    save_parameters(txt_basefilename, args)

    # ------------------------------
    # Iteration
    filename_list = []

    lya_m = lm.LyaMocks(args.seed, N_CELLS=args.ngrid, DV_KMS=args.griddv, \
        REDSHIFT_ON=not args.without_z_evo, GAUSSIAN_MOCKS=args.gauss, USE_LOG_V=not args.use_eds_v)

    if args.gauss:
        mean_flux_function = fid.meanFluxFG08
    else:
        mean_flux_function = lm.lognMeanFluxGH

    # Set up DESI observed wavelength grid
    DESI_WAVEGRID   = getDESIwavegrid(args)

    # parallel support
    dithr = int(npixels/args.nthreads)
    i1 = dithr * args.ithread
    i2 = npixels if (args.ithread == args.nthreads-1) else dithr * (1+args.ithread)

    for ipix in range(i1, i2):
        meta1 = metadata[metadata['IPIX'] == ipix]
        ntemp = len(meta1['MOCKID'])
        z_qso = meta1['Z'][:, None]

        if ntemp == 0:
            continue

        # z_center = (fid.LYA_CENTER_WVL / fid.LYA_WAVELENGTH) * (1. + z_qso) - 1
        lya_m.setCentralRedshift(3.0)

        wave, fluxes, errors = lya_m.resampledMocks(ntemp, err_per_final_pixel=args.sigma_per_pixel, \
            spectrograph_resolution=args.specres, obs_wave_centers=DESI_WAVEGRID, \
            logspacing_obswave=args.use_logspaced_wave, keep_empty_bins=args.keep_nolya_pixels)

        # Remove absorption above Lya
        nonlya_ind = wave > fid.LYA_WAVELENGTH * (1+z_qso)

        for i in range(ntemp):
            fluxes[i][nonlya_ind[i]] = 1
            errors[i][nonlya_ind[i]] = 1

        if not args.save_full_flux:
            if args.without_z_evo:
                spectrum_z = z_qso * np.ones_like(wave)
            else:
                spectrum_z = np.array(wave, dtype=np.double) / fid.LYA_WAVELENGTH - 1

            true_mean_flux = mean_flux_function(spectrum_z)

            fluxes  = fluxes / true_mean_flux - 1
            errors /= true_mean_flux

        if not args.nosave and args.save_qqfile:
            assert args.keep_nolya_pixels
            # assert not args.chunk_fixed

            P = int(ipix/100)
            dir1 = ospath_join(args.Outputdir, f"{P}")
            dir2 = ospath_join(dir1, f"{ipix}")
            os_makedirs(dir1, exist_ok=True)
            os_makedirs(dir2, exist_ok=True)
            fname = ospath_join(dir2, f"lya-transmission-{args.hp_nside}-{ipix}.fits.gz")
            
            qqfile = QQFile(fname, 'rw')
            qqfile.writeAll(meta1, wave, fluxes)

            continue

        ############
        ### CONTINUE DEVELOPEMENT HERE ON
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
            if args.chunk_dyn:
                waves, fluxes, errors = so.chunkDynamic(wave, fluxes, errors, len(wave))
            if args.chunk_fixed:
                NUMBER_OF_CHUNKS = 3
                FIXED_CHUNK_EDGES = np.linspace(fid.LYA_FIRST_WVL, fid.LYA_LAST_WVL, num=NUMBER_OF_CHUNKS+1)
                waves, fluxes, errors = so.divideIntoChunks(waves, fluxes, errors, z_qso, FIXED_CHUNK_EDGES)

            nchunks = len(waves)
            nid = meta1['MOCKID']
            fname = ["desilite_seed%d_id%d_%d_z%.1f%s.dat" \
                % (args.seed, nid[nc], nc, z_qso[nc], settings_txt) for nc in range(nchunks)]

            filename_list.extend(fname)
            elif not args.nosave:
                save_data(waves, fluxes, errors, fname, z_qso, meta1['DEC'], ra, args)

            if args.plot:
                save_plots(waves, fluxes, errors, fname, args)


    # Save the list of files in a txt
    temp_fname = ospath_join(args.Outputdir, "file_list_qso.txt") # "%s_filelist.txt" % txt_basefilename
    print("Saving chunk spectra file list as ", temp_fname)
    toWrite = open(temp_fname, 'w')
    toWrite.write("%d\n" % len(filename_list))
    for f in filename_list:
        toWrite.write(f +"\n")
    toWrite.close()






