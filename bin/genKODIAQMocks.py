#!/usr/bin/env python

# TODO
# Fix asu.tsv location
# Add --continuum option for continuum errors.
# Add --dla option to add DLAs
from os.path import join as ospath_join
from os import makedirs as os_makedirs
import argparse

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import qsotools.mocklib  as lm
import qsotools.specops  as so
from qsotools.io import BinaryQSO
import qsotools.kodiaqio as ki
import qsotools.fiducial as fid

# Define Saving Functions
# ------------------------------
def save_parameters(txt_basefilename, args):
    Parameters_txt = ("Parameters for these mocks\n"
                    "Type                 : %s\n"
                    "Velocity to Redshift : %s\n"
                    "Errors               : %s\n"
                    "Seed                 : %d\n"
                    "NGrid                : %d\n"
                    "GridPixelSize        : %f\n"
                    "LowResPixelSize      : %f\n"
                    "LyaFirst             : %.1f\n"
                    "LyaLast              : %.1f\n"
                    "DynamicChunking      : %s\n"
                    "SkipShortChunks      : %f\n"
                    "Redshift Evolution   : %s\n") % ( 
        "Gaussian Mocks" if args.gauss else "Lognormal Mocks", \
        "Logarithmic" if not args.use_eds_v else "EdS", \
        "OFF" if args.noerrors else "ON", \
        args.Seed, \
        args.ngrid, \
        args.griddv, \
        args.lowdv if args.lowdv else 0., \
        fid.LYA_FIRST_WVL, \
        fid.LYA_LAST_WVL, \
        "ON" if args.chunk_dyn else "OFF", \
        args.skip if args.skip else 0., \
        "ON" if not args.without_z_evo else "OFF")
            
    temp_fname = "%s_parameters.txt" % txt_basefilename
    print("Saving parameteres to", temp_fname)
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()

def save_plots(wch, fch, ech, fnames, obs_fits):
    fig_title = "%s/%s/%s at z=%.2f" \
    % (obs_fits.qso_name, obs_fits.pi_date, obs_fits.spec_prefix, obs_fits.z_qso)

    for (f, e, fname) in zip(fch, ech, fnames):
        plt.cla()
        plt.figure(figsize=(8,4), dpi=300)
        plt.title(fig_title)
        plt.plot(wch, f, 'b-')
        plt.plot(wch, e, 'r-')
        plt.grid(True, "major")
        plt.savefig(ospath_join(args.Outputdir, "plots", fname[:-3]+"png"), bbox_inches='tight')

def save_data(waves, fluxes, errors, fnames, obs_fits, spec_res, pixel_width):
    for (w, f, e, fname) in zip(waves, fluxes, errors, fnames):
        mfile = BinaryQSO(ospath_join(args.Outputdir, fname), 'w')
        mfile.save(w, f, e, len(w), obs_fits.z_qso, obs_fits.DECL, obs_fits.RA, obs_fits.S2N, \
            spec_res, pixel_width)

def save_list_byline(array, fname):
    toWrite = open(fname, 'w')
    toWrite.write('%d\n'%len(array))
    for a in array:
        toWrite.write('%s\n'%str(a))
    toWrite.close()
# ------------------------------

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("KODIAQdir", help="Directory of KODIAQ")
    parser.add_argument("Outputdir", help="Output directory")
    parser.add_argument("Seed", help="Seed to generate random numbers.", type=int)

    parser.add_argument("--asu-path", help="Table containing KODIAQ qso list.", default=ki.TABLE_KODIAQ_ASU)
    parser.add_argument("--save_full_flux", action="store_true", \
        help="When passed saves flux instead of fluctuations around truth.")

    # parser.add_argument("--nchunks", help="Number of chunks to divide Ly a forest. Default is 1.", \
    #   type=int, default=1)
    parser.add_argument("--chunk-dyn",  action="store_true", \
        help="Dynamic chunking splits a spectrum into three chunks if l>L/2 or into two chunks if l>L/3.")
    parser.add_argument("--kodiaq-errors", help=("Add exact KODIAQ errors onto final grid. "\
        "Beware of resampling."), action="store_true")
    parser.add_argument("--noerrors", help="Generate error free mocks", action="store_true")

    parser.add_argument("--skip", help="Skip short chunks lower than given ratio", type=float)
    parser.add_argument("--gauss", help="Generate Gaussian mocks", action="store_true")
    parser.add_argument("--without_z_evo", help="Turn of redshift evolution", action="store_true")
    parser.add_argument("--lowdv", help="Resamples grid to this pixel size (km/s) when passed", \
        type=float)
    
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=1.7)
    parser.add_argument("--z-forest-max", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=3.9)
    
    parser.add_argument("--nosave", help="Does not save mocks to output when passed", \
        action="store_true")
    parser.add_argument("--plot", help="Saves plots to output when passed", action="store_true")
    
    parser.add_argument("--use_eds_v", action="store_true", \
        help="Use EdS wavelength grid. Default is False (i.e. Log spacing).")
    parser.add_argument("--ngrid", help="Number of grid points. Default is 2^20", type=int, \
        default=2**20)
    parser.add_argument("--griddv", help="Pixel size of the grid in km/s. Default is 0.4333", \
        type=float, default=1.3/3)
    args = parser.parse_args()
    
    # Initialize lists
    no_lya_quasar_list = []

    selected_qso_list = []
    selected_qso_observation_pi = []
    selected_qso_pixwidth = []
    selected_qso_zem_list = []
    selected_qso_DR = []
    selected_qso_s2n_lya = []
    selected_qso_resampled_npixels = []
    selected_qso_specres_list = []

    # Pick mean flux function
    if args.gauss:
        mean_flux_function = fid.meanFluxFG08
    else:
        mean_flux_function = lm.lognMeanFluxGH

    # Decide if it's real data
    if "mock0/data" in args.Outputdir:
        print("MOCK0 is real data!")
        REAL_DATA = True
        mean_flux_function = fid.meanFluxFG08
    else:
        REAL_DATA = False

    # Set settings text
    settings_txt  = '_gaussian' if args.gauss else '_lognormal' 
    settings_txt += '_dv%.1f' % args.lowdv if args.lowdv else ''
    settings_txt += '_noz' if args.without_z_evo else ''

    txt_basefilename  = "%s/kodiaq%s" % (args.Outputdir, settings_txt)

    save_parameters(txt_basefilename, args)
    # ------------------------------

    # Set up initial objects and variables
    filename_list = []
    specres_list  = set()

    qso_iter = ki.KODIAQ_QSO_Iterator(args.KODIAQdir, args.asu_path, clean_pix=False)
    
    if not REAL_DATA:
        lya_m = lm.LyaMocks(args.Seed, N_CELLS=args.ngrid, DV_KMS=args.griddv, \
            REDSHIFT_ON=not args.without_z_evo, \
            GAUSSIAN_MOCKS=args.gauss, USE_LOG_V=not args.use_eds_v)

    # Create/Check directory
    os_makedirs(args.Outputdir, exist_ok=True)
    if args.plot:
        os_makedirs(ospath_join(args.Outputdir, "plots"), exist_ok=True)

    # Start iterating quasars in KODIAQ sample
    # Each quasar has multiple observations
    # Pick the one with highest signal to noise in Ly-alpha region
    for qso in qso_iter:
        print("********************************************", flush=True)
        obs_iter = ki.KODIAQ_OBS_Iterator(qso)

        # Pick highest S2N obs
        max_obs_spectrum, maxs2n = obs_iter.maxLyaObservation()
        max_obs_spectrum.print_details()

        if maxs2n == -1:
            print("SKIP: No Lya coverage!")
            no_lya_quasar_list.append(qso.qso_name)
            continue

        z_center = (fid.LYA_CENTER_WVL / fid.LYA_WAVELENGTH) * (1. + qso.z_qso) - 1
        print("Ly-alpha forest central redshift is ", z_center)

        pixel_width  = args.lowdv if args.lowdv else max_obs_spectrum.dv
        low_spec_res = max_obs_spectrum.specres
        MAX_NO_PIXELS = int(fid.LIGHT_SPEED * np.log(fid.LYA_LAST_WVL/fid.LYA_FIRST_WVL) / pixel_width)

        print("Spectral Res: from %d to %d." % (max_obs_spectrum.specres, low_spec_res))
        print("Pixel width: from %.2f to %.2f km/s" %(max_obs_spectrum.dv, pixel_width))

        specres_list.add(low_spec_res)

        if not REAL_DATA:
            lya_m.setCentralRedshift(z_center)

            # Decide error on final pixels
            final_error = 0 if args.noerrors or args.kodiaq_errors else 0.1

            wave, fluxes, errors = lya_m.resampledMocks(1, err_per_final_pixel=final_error, \
                spectrograph_resolution=low_spec_res, resample_dv=args.lowdv, \
                obs_wave_centers=max_obs_spectrum.wave)
        else:
            max_obs_spectrum.applyMask()
            print("Number of pixel in original resolution for the entire spectrum is %d."%max_obs_spectrum.N)
            
            # Re-sample real data onto lower resolution grid
            if args.lowdv:
                wave, fluxes, errors = so.resample(max_obs_spectrum.wave, \
                    max_obs_spectrum.flux.reshape(1,max_obs_spectrum.N), \
                    max_obs_spectrum.error.reshape(1,max_obs_spectrum.N), pixel_width)
                print("Number of pixel in lower resolution (%.2f km/s) for the entire spectrum is %d."\
                    %(pixel_width, len(wave)))
            else:
                wave, fluxes, errors = max_obs_spectrum.wave, \
                max_obs_spectrum.flux.reshape(1,max_obs_spectrum.N), \
                max_obs_spectrum.error.reshape(1,max_obs_spectrum.N)

        # Cut Lyman-alpha forest region
        lyman_alpha_ind = np.logical_and(wave >= fid.LYA_FIRST_WVL * (1+qso.z_qso), \
            wave <= fid.LYA_LAST_WVL * (1+qso.z_qso))
        # Cut analysis boundaries
        forest_boundary = np.logical_and(wave >= fid.LYA_WAVELENGTH*(1+args.z_forest_min), \
            wave <= fid.LYA_WAVELENGTH*(1+args.z_forest_max))
        lyman_alpha_ind = np.logical_and(lyman_alpha_ind, forest_boundary)

        wave   = wave[lyman_alpha_ind]
        fluxes = np.array([f[lyman_alpha_ind] for f in fluxes])
        errors = np.array([e[lyman_alpha_ind] for e in errors])

        if args.without_z_evo:
            spectrum_z = z_center * np.ones_like(wave)
        else:
            spectrum_z = np.array(wave, dtype=np.double)  / fid.LYA_WAVELENGTH - 1

        if not args.save_full_flux:
            true_mean_flux = mean_flux_function(spectrum_z)

            fluxes  = fluxes / true_mean_flux - 1
            errors /= true_mean_flux

        # Save information about selected quasars
        selected_qso_list.append(qso.qso_name)
        selected_qso_observation_pi.append(max_obs_spectrum.pi_date)
        selected_qso_pixwidth.append(max_obs_spectrum.dv)
        selected_qso_zem_list.append(qso.z_qso)
        selected_qso_specres_list.append(max_obs_spectrum.specres)
        # selected_qso_DR.append(readme_table[IND_PICKED]['kodrelease'])
        selected_qso_s2n_lya.append(maxs2n)
        selected_qso_resampled_npixels.append(len(wave))
        
        # Skip short spectrum
        if args.skip and len(wave) < MAX_NO_PIXELS * args.skip:
            print("This spectrum has few points.")
            continue

        print("Lowest Obs Wave, data-mock", max_obs_spectrum.wave[0], wave[0])
        print("Highest Obs Wave, data-mock", max_obs_spectrum.wave[-1], wave[-1])
        
        if args.chunk_dyn:
            wave, fluxes, errors, nchunks = so.chunkDynamic(wave, fluxes[0], errors[0], MAX_NO_PIXELS)

            temp_fname = ["%s_%s_%s-%d_%dA_%dA%s.dat" % (qso.qso_name, max_obs_spectrum.pi_date, \
                max_obs_spectrum.spec_prefix, nc, wave[nc][0], wave[nc][-1], settings_txt) \
                for nc in range(nchunks)]
        else:
            wave  = [wave]
            temp_fname = ["%s_%s_%s_%dA_%dA%s.dat" % (qso.qso_name, max_obs_spectrum.pi_date, \
                max_obs_spectrum.spec_prefix, wave[0][0], wave[0][-1], settings_txt)]
            
        filename_list.extend(temp_fname) 

        # if args.plot:
        #     save_plots(wch, fch, ech, temp_fname, max_obs_spectrum)

        if not args.nosave:
            save_data(wave, fluxes, errors, temp_fname, max_obs_spectrum, low_spec_res, pixel_width)
    # ------------------------------

    temp_fname = ospath_join(args.Outputdir, "specres_list.txt" )
    print("Saving spectral resolution values as ", temp_fname)
    save_list_byline(specres_list, temp_fname)

    # Save the list of files in a txt
    temp_fname = ospath_join(args.Outputdir, "file_list_qso.txt")
    #temp_fname = "%s/_filelist.txt" % txt_basefilename
    print("Saving chunk spectra file list as ", temp_fname)
    save_list_byline(filename_list, temp_fname)






