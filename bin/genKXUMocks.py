#!/usr/bin/env python

from os.path import join as ospath_join
from os import makedirs as os_makedirs
import glob
import argparse

import numpy as np

import qsotools.mocklib  as lm
import qsotools.specops  as so
from qsotools.io import BinaryQSO, XQ100Fits, SQUADFits, \
    TABLE_KODIAQ_ASU, KODIAQ_QSO_Iterator, KODIAQ_OBS_Iterator
import qsotools.fiducial as fid

# Define Saving Functions
# ------------------------------
def saveParameters(txt_basefilename, f1, f2, args):
    err_txt = str(args.const_error) if args.const_error \
        else "Observed Errors"

    if args.chunk_dyn:
        chn_txt = "Dynamic"
    elif args.chunk_fixed:
        chn_txt = "Fixed"
    else:
        chn_txt = "OFF"

    Parameters_txt = ("Parameters for these mocks\n"
                    "Type                 : %s\n"
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
        err_txt, \
        args.seed, \
        args.ngrid, \
        args.griddv, \
        args.lowdv if args.lowdv else 0., \
        f1, \
        f2, \
        chn_txt, \
        args.skip if args.skip else 0., \
        "ON" if not args.without_z_evo else "OFF")
            
    temp_fname = "%s_parameters.txt" % txt_basefilename
    print("Saving parameteres to", temp_fname)
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()

def saveData(waves, fluxes, errors, fnames, obs_fits, spec_res, pixel_width, args):
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

def convert2DeltaFlux(wave, fluxes, errors, mean_flux_function, args):
    if args.without_z_evo:
        spectrum_z = z_center * np.ones_like(wave)
    else:
        spectrum_z = np.array(wave, dtype=np.double)  / fid.LYA_WAVELENGTH - 1

    if not args.save_full_flux:
        true_mean_flux = mean_flux_function(spectrum_z)

        fluxes  = fluxes / true_mean_flux - 1
        errors /= true_mean_flux

    return fluxes, errors

def genMocks(qso, f1, f2, mean_flux_function, specres_list, \
    isRealData, mean_flux_hist, args, disableChunk=False):
    forest_c = (f1+f2)/2
    z_center = (forest_c / fid.LYA_WAVELENGTH) * (1. + qso.z_qso) - 1
    print("Ly-alpha forest central redshift is ", z_center)

    pixel_width  = args.lowdv if args.lowdv else qso.dv
    low_spec_res = qso.specres
    MAX_NO_PIXELS = int(fid.LIGHT_SPEED * np.log(fid.LYA_LAST_WVL/fid.LYA_FIRST_WVL) / pixel_width)
    
    print("Number of pixel in original resolution for the entire spectrum is %d."%qso.size)

    qso.setOutliersMask(sigma=2.5)
    if args.mask_spikes_zscore:
        qso.setZScoreMask(args.mask_spikes_zscore)
    
    # This sets err=1e10 and flux=0
    qso.applyMask(removePixels=False)

    if not isRealData:
        lya_m.setCentralRedshift(z_center)
        if args.const_resolution:
            low_spec_res = args.const_resolution

        qso = lya_m.qsoMock(qso, low_spec_res, args.const_error)

    print("Spectral Res: from %d to %d." % (qso.specres, low_spec_res))
    print("Pixel width: from %.2f to %.2f km/s" %(qso.dv, pixel_width))
    
    qso.applyMask(good_pixels=qso.error<1e6, removePixels=not args.keep_masked_pix)
    if args.mask_dlas:
        qso.applyMaskDLAs(removePixels=not args.keep_masked_pix)

    if args.compute_mean_flux:
        mean_flux_hist.addSpectrum(qso, f1, f2)

    qso.cutForestAnalysisRegion(f1, f2, args.z_forest_min, args.z_forest_max)

    # Re-sample real data onto lower resolution grid
    if args.lowdv:
        wave, fluxes, errors = so.resample(qso.wave, qso.flux.reshape(1,qso.size), \
            qso.error.reshape(1,qso.size), pixel_width)
        print("Number of pixel in lower resolution (%.2f km/s) for the entire spectrum is %d."\
            %(pixel_width, len(wave)))
    else:
        wave, fluxes, errors = qso.wave, qso.flux.reshape(1,qso.size), qso.error.reshape(1,qso.size)

    fluxes, errors = convert2DeltaFlux(wave, fluxes, errors, mean_flux_function, args)

    # Skip short spectrum
    isShort = lambda x: (args.skip and len(x) < MAX_NO_PIXELS * args.skip) or len(x)==0
    if isShort(wave):
        raise ValueError("Short spectrum", len(wave), MAX_NO_PIXELS)

    specres_list.add((low_spec_res, pixel_width))
    print("Lowest Obs Wave, data: %.3f - mock: %.3f"%(qso.wave[0], wave[0]))
    print("Highest Obs Wave, data: %.3f - mock: %.3f"%(qso.wave[-1], wave[-1]))

    if not disableChunk and args.chunk_dyn:
        wave, fluxes, errors = so.chunkDynamic(wave, fluxes[0], errors[0], MAX_NO_PIXELS)
    elif not disableChunk and args.chunk_fixed:
        NUMBER_OF_CHUNKS = 3
        FIXED_CHUNK_EDGES = np.linspace(forest_1, forest_2, num=NUMBER_OF_CHUNKS+1)
        wave, fluxes, errors = so.divideIntoChunks(wave, fluxes[0], errors[0], \
            qso.z_qso, FIXED_CHUNK_EDGES)
    else:
        wave = [wave]

    wave   = [x for x in wave   if not isShort(x)]
    fluxes = [x for x in fluxes if not isShort(x)]
    errors = [x for x in errors if not isShort(x)]

    return wave, fluxes, errors, low_spec_res, pixel_width

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("OutputDir", help="Output directory")
    parser.add_argument("--seed", help="Seed to generate random numbers.", type=int, default=68970)

    parser.add_argument("--KODIAQdir", help="Directory of KODIAQ")
    
    parser.add_argument("--XQ100Dir", help="Directory of XQ100")
    parser.add_argument("--UVESSQUADDir", help="Directory of SQUAD/UVES")

    parser.add_argument("--save_full_flux", action="store_true", \
        help="When passed saves flux instead of fluctuations around truth.")

    parser.add_argument("--chunk-dyn",  action="store_true", \
        help="Splits spectrum into three chunks if n>2N/3 or into two chunks if n>N/3.")
    parser.add_argument("--chunk-fixed",  action="store_true", \
        help="Splits spectrum into 3 chunks at fixed rest frame wavelengths")

    parser.add_argument("--skip", help="Skip short chunks lower than given ratio", type=float)

    parser.add_argument("--const-resolution", type=int, \
        help="Use this resolution for mocks when passed instead.")
    parser.add_argument("--const-error", type=float, \
        help="Use this constant error for mocks when passed instead.")
    parser.add_argument("--observed-errors", help=("Add exact KODIAQ/XQ-100 errors onto final grid. "\
        "Beware of resampling."), action="store_true")

    parser.add_argument("--gauss", help="Generate Gaussian mocks", action="store_true")
    parser.add_argument("--without_z_evo", help="Turn of redshift evolution", action="store_true")
    parser.add_argument("--lowdv", help="Resamples grid to this pixel size (km/s) when passed", \
        type=float)
    parser.add_argument("--mask-dlas", action="store_true")
    parser.add_argument("--mask-spikes-zscore", help="Mask spikes by given zscore.", type=float)
    parser.add_argument("--keep-masked-pix", help="Assign large errors.", action="store_true")
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=1.7)
    parser.add_argument("--z-forest-max", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=4.3)
    
    parser.add_argument("--side-band", type=int, default=0, help="Side band. Default: %(default)s")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--compute-mean-flux", action="store_true")
    parser.add_argument("--nosave", help="Does not save mocks to output when passed", \
        action="store_true")
    
    parser.add_argument("--ngrid", help="Number of grid points. Default is 2^20", type=int, \
        default=2**20)
    parser.add_argument("--griddv", help="Pixel size of the grid in km/s. Default is 0.4333", \
        type=float, default=1.3/3)
    args = parser.parse_args()

    # Create/Check directory
    os_makedirs(args.OutputDir, exist_ok=True)

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

    # Pick mean flux function
    if args.gauss:
        mean_flux_function = fid.meanFluxFG08
    else:
        mean_flux_function = lm.lognMeanFluxGH

    # Decide if it's real data
    if "mock0/data" in args.OutputDir or args.real_data:
        print("MOCK0 is real data!")
        isRealData = True
        settings_txt  = ''
    else:
        isRealData = False
        settings_txt  = '_gaussian' if args.gauss else '_lognormal' 

    # Set settings text
    settings_txt += '_dv%.1f' % args.lowdv if args.lowdv else ''
    settings_txt += '_noz' if args.without_z_evo else ''
    settings_txt += '_sb%d'%args.side_band if args.side_band else ''
    settings_txt += '_masked-dlas' if args.mask_dlas else ''

    txt_basefilename  = "%s/highres%s" % (args.OutputDir, settings_txt)

    saveParameters(txt_basefilename, forest_1, forest_2, args)
    # ------------------------------

    # Set up initial objects and variables
    no_lya_quasar_list = []
    filename_list = []
    specres_list  = set()
    
    if not isRealData:
        lya_m = lm.LyaMocks(args.seed, N_CELLS=args.ngrid, DV_KMS=args.griddv, \
            REDSHIFT_ON=not args.without_z_evo, GAUSSIAN_MOCKS=args.gauss)

    # ------------------------------    
    # Start with KODIAQ
    if args.KODIAQdir:
        print("RUNNING ON KODIAQ.........")
        qso_iter = KODIAQ_QSO_Iterator(args.KODIAQdir, clean_pix=False)

        if isRealData:
            mean_flux_function = fid.meanFluxFG08

        kod_mf_hist = so.MeanFluxHist(args.z_forest_min, args.z_forest_max)

        # Start iterating quasars in KODIAQ sample
        # Each quasar has multiple observations
        # Pick the one with highest signal to noise in Ly-alpha region
        for qso in qso_iter:
            print("********************************************", flush=True)
            obs_iter = KODIAQ_OBS_Iterator(qso)

            # Pick highest S2N obs
            max_obs_spectrum, maxs2n = obs_iter.maxLyaObservation(forest_1, forest_2)
            max_obs_spectrum.print_details()

            if maxs2n == -1:
                print("SKIP: No Lya or Side Band coverage!")
                no_lya_quasar_list.append(qso.qso_name)
                continue

            try:
                wave, fluxes, errors, lspecr, pixw = genMocks(max_obs_spectrum, \
                    forest_1, forest_2, mean_flux_function, specres_list, \
                    isRealData, kod_mf_hist, args)
            except ValueError as ve:
                # print(ve)
                print(ve.args)
                continue
            
            nchunks = len(wave)
            temp_fname = ["k%s_%s_%s-%d_%dA_%dA%s.dat" % (qso.qso_name, max_obs_spectrum.pi_date, \
                max_obs_spectrum.spec_prefix, nc, wave[nc][0], wave[nc][-1], settings_txt) \
                for nc in range(nchunks)]
                
            filename_list.extend(temp_fname) 

            if not args.nosave:
                saveData(wave, fluxes, errors, temp_fname, max_obs_spectrum, lspecr, pixw, args)

        if args.compute_mean_flux:
            kod_mf_hist.getMeanFlux()
            kod_mf_hist.saveHistograms(ospath_join(args.OutputDir, "kod-stats%s"%settings_txt))

    # ------------------------------
    # XQ-100
    if args.XQ100Dir:
        print("RUNNING ON XQ-100.........")
        if isRealData:
            mean_flux_function = lambda z: fid.evaluateBecker13MeanFlux(z, *fid.XQ100_FIT_PARAMS)

        xq_mf_hist = so.MeanFluxHist(args.z_forest_min, args.z_forest_max)

        for f in glob.glob(ospath_join(args.XQ100Dir, "*.fits")):
            print("********************************************", flush=True)
            qso = XQ100Fits(f, correctSeeing=True)
            qso.getS2NLya(forest_1, forest_2)

            if qso.s2n_lya == -1:
                print("SKIP: No Lya or Side Band coverage!")
                no_lya_quasar_list.append(f)
                continue

            try:
                wave, fluxes, errors, lspecr, pixw = genMocks(qso, forest_1, \
                    forest_2, mean_flux_function, specres_list, \
                    isRealData, xq_mf_hist, args, disableChunk=True)
            except ValueError as ve:
                # print(ve)
                print(ve.args)
                continue
            
            temp_fname = ["xq%s_%s_%dA_%dA%s.dat" % (qso.object.replace(" ", ""), qso.arm, \
                wave[0][0], wave[0][-1], settings_txt)]
                
            filename_list.extend(temp_fname) 

            if not args.nosave:
                saveData(wave, fluxes, errors, temp_fname, qso, lspecr, pixw, args)

        if args.compute_mean_flux:
            xq_mf_hist.getMeanFlux()
            xq_mf_hist.saveHistograms(ospath_join(args.OutputDir, "xq-stats%s"%settings_txt))

    # ------------------------------
    # UVES/SQUAD
    if args.UVESSQUADDir:
        print("RUNNING ON SQUAD/UVES.........")

        if isRealData:
            mean_flux_function = lambda z: fid.evaluateBecker13MeanFlux(z, *fid.UVES_FIT_PARAMS_NODLA)

        us_mf_hist = so.MeanFluxHist(args.z_forest_min, args.z_forest_max)

        for f in glob.glob(ospath_join(args.UVESSQUADDir, "*.fits")):
            print("********************************************", flush=True)
            qso = SQUADFits(f, correctSeeing=True, corrError=True)
            qso.getS2NLya(forest_1, forest_2)

            if qso.s2n_lya == -1:
                print("SKIP: No Lya or Side Band coverage!")
                no_lya_quasar_list.append(f)
                continue

            if qso.flag != '0':
                print("SKIP: Spec. status is not 0.")
                no_lya_quasar_list.append(f)
                continue
                
            try:
                wave, fluxes, errors, lspecr, pixw = genMocks(qso, forest_1, forest_2, \
                    mean_flux_function, specres_list, isRealData, us_mf_hist, args)
            except ValueError as ve:
                # print(ve)
                print(ve.args)
                continue
            
            nchunks = len(wave)
            temp_fname = ["us%s_%d_w%d-%dA%s.dat" % (qso.object.replace(" ", ""), nc, \
                    wave[nc][0], wave[nc][-1], settings_txt) for nc in range(nchunks)]
                
            filename_list.extend(temp_fname) 

            if not args.nosave:
                saveData(wave, fluxes, errors, temp_fname, qso, lspecr, pixw, args)

        if args.compute_mean_flux:
            us_mf_hist.getMeanFlux()
            us_mf_hist.saveHistograms(ospath_join(args.OutputDir, "us-stats%s"%settings_txt))

    temp_fname = ospath_join(args.OutputDir, "specres_list.txt" )
    print("Saving spectral resolution values as ", temp_fname)
    saveListByLine(specres_list, temp_fname)

    # Save the list of files in a txt
    temp_fname = ospath_join(args.OutputDir, "file_list_qso.txt")
    #temp_fname = "%s/_filelist.txt" % txt_basefilename
    print("Saving chunk spectra file list as ", temp_fname)
    saveListByLine(filename_list, temp_fname)


