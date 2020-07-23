#!/usr/bin/env python

from os.path import join as ospath_join
from os import makedirs as os_makedirs
from os import listdir as os_listdir
import argparse

import numpy as np

import qsotools.mocklib  as lm
import qsotools.specops  as so
from qsotools.io import BinaryQSO
from qsotools.xq100io import XQ100Fits
import qsotools.fiducial as fid

# Define Saving Functions
# ------------------------------
def saveParameters(txt_basefilename, args):
    Parameters_txt = ("Parameters for these mocks\n"
                    "Type                 : %s\n"
                    "Errors               : %s\n"
                    "Seed                 : %d\n"
                    "NGrid                : %d\n"
                    "GridPixelSize        : %f\n"
                    "LowResPixelSize      : %f\n"
                    "LyaFirst             : %.1f\n"
                    "LyaLast              : %.1f\n"
                    "SkipShortChunks      : %f\n"
                    "Redshift Evolution   : %s\n") % ( 
        "Gaussian Mocks" if args.gauss else "Lognormal Mocks", \
        "OFF" if args.noerrors else "ON", \
        args.Seed, \
        args.ngrid, \
        args.griddv, \
        args.lowdv if args.lowdv else 0., \
        fid.LYA_FIRST_WVL, \
        fid.LYA_LAST_WVL, \
        args.skip if args.skip else 0., \
        "ON" if not args.without_z_evo else "OFF")
            
    temp_fname = "%s_parameters.txt" % txt_basefilename
    print("Saving parameteres to", temp_fname)
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()

def saveData(waves, fluxes, errors, fnames, obs_fits, spec_res, pixel_width):
    for (w, f, e, fname) in zip(waves, fluxes, errors, fnames):
        mfile = BinaryQSO(ospath_join(args.OutputDir, fname), 'w')
        mfile.save(w, f, e, len(w), obs_fits.z_qso, obs_fits.dec, obs_fits.ra, obs_fits.s2n, \
            spec_res, pixel_width)

def saveListByLine(array, fname):
    toWrite = open(fname, 'w')
    toWrite.write('%d\n'%len(array))
    for a in array:
        if len(a) == 2:
            toWrite.write("%d %.1f\n"%(a[0], a[1]))
        else:
            toWrite.write('%s\n'%str(a))
    toWrite.close()
# ------------------------------

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("XQ100Dir", help="Directory of XQ100")
    parser.add_argument("OutputDir", help="Output directory")
    parser.add_argument("Seed", help="Seed to generate random numbers.", type=int)

    parser.add_argument("--save_full_flux", action="store_true", \
        help="When passed saves flux instead of fluctuations around truth.")

    parser.add_argument("--xq100-errors", help=("Add exact XQ100 errors onto final grid. "\
        "Beware of resampling."), action="store_true")
    parser.add_argument("--noerrors", help="Generate error free mocks", action="store_true")

    parser.add_argument("--skip", help="Skip short chunks lower than given ratio", type=float)
    parser.add_argument("--gauss", help="Generate Gaussian mocks", action="store_true")
    parser.add_argument("--without_z_evo", help="Turn of redshift evolution", action="store_true")
    parser.add_argument("--lowdv", help="Resamples grid to this pixel size (km/s) when passed", \
        type=float)
    
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=2.9)
    parser.add_argument("--z-forest-max", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=4.3)
    
    parser.add_argument("--side-band", type=int, default=0, help="Side band. Default: %(default)s")

    parser.add_argument("--nosave", help="Does not save mocks to output when passed", \
        action="store_true")
   
    parser.add_argument("--ngrid", help="Number of grid points. Default is 2^18", type=int, \
        default=2**18)
    parser.add_argument("--griddv", help="Pixel size of the grid in km/s. Default: %(default)s", \
        type=float, default=1.)
    args = parser.parse_args()

    # Initialize waverange
    if args.side_band == 0:
        forest_1 = fid.LYA_FIRST_WVL
        forest_2 = fid.LYA_LAST_WVL
    elif args.side_band == 1:
        forest_1 = fid.Si4_FIRST_WVL
        forest_2 = fid.Si4_LAST_WVL
    elif args.side_band == 2:
        forest_1 = fid.C4_FIRST_WVL
        forest_2 = fid.C4_LAST_WVL
    
    forest_c = (forest_1+forest_2)/2

    # Pick mean flux function
    if args.gauss:
        mean_flux_function = fid.meanFluxFG08
    else:
        mean_flux_function = lm.lognMeanFluxGH

    # Decide if it's real data
    if "mock0/data" in args.OutputDir:
        print("MOCK0 is real data!")
        REAL_DATA = True
        mean_flux_function = lambda z: fid.evaluateBecker13MeanFlux(z, *fid.XQ100_FIT_PARAMS)
        settings_txt  = ''
    else:
        REAL_DATA = False
        settings_txt  = '_gaussian' if args.gauss else '_lognormal' 

    # Set settings text
    settings_txt += '_dv%.1f' % args.lowdv if args.lowdv else ''
    settings_txt += '_noz' if args.without_z_evo else ''

    txt_basefilename  = "%s/xq100%s" % (args.OutputDir, settings_txt)

    saveParameters(txt_basefilename, args)
    # ------------------------------

    # Set up initial objects and variables
    no_lya_quasar_list = []
    selected_qso_list  = []
    filename_list = []
    specres_list  = set()
    
    if not REAL_DATA:
        lya_m = lm.LyaMocks(args.Seed, N_CELLS=args.ngrid, DV_KMS=args.griddv, \
            REDSHIFT_ON=not args.without_z_evo, GAUSSIAN_MOCKS=args.gauss)

    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)

    for f in os_listdir(args.XQ100Dir):
        if not f.endswith(".fits"):
            continue
        print("********************************************", flush=True)
        qso = XQ100Fits(ospath_join(args.XQ100Dir, f))
        qso.getS2NLya();

        if qso.s2n_lya == -1:
            print("SKIP: No Lya coverage!")
            no_lya_quasar_list.append(f)
            continue

        z_center = (forest_c / fid.LYA_WAVELENGTH) * (1. + qso.z_qso) - 1
        print("Ly-alpha forest central redshift is ", z_center)

        pixel_width  = args.lowdv if args.lowdv else qso.dv
        low_spec_res = qso.specres
        MAX_NO_PIXELS = int(fid.LIGHT_SPEED * np.log(fid.LYA_LAST_WVL/fid.LYA_FIRST_WVL) / pixel_width)

        print("Spectral Res: from %d to %d." % (qso.specres, low_spec_res))
        print("Pixel width: from %.2f to %.2f km/s" %(qso.dv, pixel_width))

        specres_list.add((low_spec_res, pixel_width))

        if not REAL_DATA:
            lya_m.setCentralRedshift(z_center)

            # Decide error on final pixels
            final_error = 0 if args.noerrors or args.xq100_errors else 0.05

            wave, fluxes, errors = lya_m.resampledMocks(1, err_per_final_pixel=final_error, \
                spectrograph_resolution=low_spec_res, resample_dv=args.lowdv, \
                obs_wave_centers=qso.wave)
        else:
            qso.maskOutliers()
            qso.applyMask()
            print("Number of pixel in original resolution for the entire spectrum is %d."%qso.size)
            
            # Re-sample real data onto lower resolution grid
            if args.lowdv:
                wave, fluxes, errors = so.resample(qso.wave, qso.flux.reshape(1,qso.size), \
                    qso.error.reshape(1,qso.size), pixel_width)
                print("Number of pixel in lower resolution (%.2f km/s) for the entire spectrum is %d."\
                    %(pixel_width, len(wave)))
            else:
                wave, fluxes, errors = qso.wave, qso.flux.reshape(1,qso.size), qso.error.reshape(1,qso.size)

        # Cut Lyman-alpha forest region
        lyman_alpha_ind = np.logical_and(wave >= forest_1 * (1+qso.z_qso), \
            wave <= forest_2 * (1+qso.z_qso))
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

        # Skip short spectrum
        if (args.skip and len(wave) < MAX_NO_PIXELS * args.skip) or len(wave)==0:
            print("This spectrum has few points.")
            continue

        # Save information about selected quasars
        selected_qso_list.append(f)

        print("Lowest Obs Wave, data-mock", qso.wave[0], wave[0])
        print("Highest Obs Wave, data-mock", qso.wave[-1], wave[-1])
        
        wave  = [wave]
        temp_fname = ["%s_%s_%dA_%dA%s.dat" % (qso.object.replace(" ", ""), qso.arm, \
            wave[0][0], wave[0][-1], settings_txt)]
            
        filename_list.extend(temp_fname) 

        if not args.nosave:
            saveData(wave, fluxes, errors, temp_fname, qso, low_spec_res, pixel_width)
    # ------------------------------

    temp_fname = ospath_join(args.OutputDir, "specres_list.txt" )
    print("Saving spectral resolution values as ", temp_fname)
    saveListByLine(specres_list, temp_fname)

    # Save the list of files in a txt
    temp_fname = ospath_join(args.OutputDir, "file_list_qso.txt")
    #temp_fname = "%s/_filelist.txt" % txt_basefilename
    print("Saving chunk spectra file list as ", temp_fname)
    saveListByLine(filename_list, temp_fname)






