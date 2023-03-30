#!/usr/bin/env python

from os.path import join as ospath_join
from os import makedirs as os_makedirs
import glob
import argparse

import numpy as np

import qsotools.mocklib  as lm
import qsotools.specops  as so
import qsotools.io as qio
import qsotools.fiducial as fid

# Global variables
global filename_list
global specres_list
global spectral_record_list
global lya_m

filename_list = []
specres_list  = set()
spectral_record_list = qio.SpectralRecordList()

ERROR_EPS_CUT = 1e-6

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
    print("Saving parameteres to", temp_fname, flush=True)
    toWrite = open(temp_fname, 'w')
    toWrite.write(Parameters_txt)
    toWrite.close()

def saveData(waves, fluxes, errors, fnames, obs_fits, spec_res, pixel_width, args):
    for (w, f, e, fname) in zip(waves, fluxes, errors, fnames):
        mfile = qio.BinaryQSO(ospath_join(args.OutputDir, fname), 'w')
        mfile.save(w, f, e, len(w), obs_fits.z_qso, obs_fits.coord.dec.rad, obs_fits.coord.ra.rad, \
            obs_fits.s2n, spec_res, pixel_width)

# ------------------------------
def convert2DeltaFlux(wave, fluxes, errors, meanFluxFunc, args):
    if not args.save_full_flux:
        if args.without_z_evo:
            spectrum_z = z_center * np.ones_like(wave)
        else:
            spectrum_z = np.array(wave, dtype=np.double) / fid.LYA_WAVELENGTH - 1
        
        true_mean_flux = meanFluxFunc(spectrum_z)

        fluxes  = fluxes / true_mean_flux - 1
        errors /= true_mean_flux

    return fluxes, errors

def safeResample(qso, lowdv, keep_masked_pix=False):
    wave, fluxes, errors = so.resample(qso.wave, qso.flux.reshape(1,qso.size), \
            qso.error.reshape(1,qso.size), lowdv)
    
    qso.dv    = lowdv
    qso.wave  = wave
    qso.flux  = fluxes[0]
    qso.error = errors[0]
    qso.mask  = np.logical_and(qso.error>ERROR_EPS_CUT, qso.error<10)
    qso.size  = qso.wave.size
    
    qso.applyMask(removePixels=not keep_masked_pix)

    return qso

def cleanup(qso, f1, f2, meanFluxFunc, args):
    forest_c = (f1+f2)/2
    z_center = (forest_c / fid.LYA_WAVELENGTH) * (1. + qso.z_qso) - 1

    if args.mask_sigma_percentile:
        qso.setOutliersMask(args.mask_sigma_percentile)
    if args.mask_spikes_zscore:
        qso.setZScoreMask(fsigma=1, esigma=args.mask_spikes_zscore)

    # This sets err=1e10 and flux=0
    qso.applyMask(removePixels=False)

    if not args.real_data:
        lya_m.setCentralRedshift(z_center)
        mock_spec_res = args.const_resolution if args.const_resolution else qso.specres

        qso = lya_m.qsoMock(qso, mock_spec_res, args.const_error)

    qso.applyMask(good_pixels=np.logical_and(qso.error>ERROR_EPS_CUT, qso.error<10), \
        removePixels=not args.keep_masked_pix)

    # manageDLAs(qso, meanFluxFunc, args)
    if args.mask_dlas:
        qso.applyMaskDLAs(removePixels=not args.keep_masked_pix)

    # If computing continuum power, set F to be C, so that it's resampled.
    # Do not set error here, because later removal relies on error < 10.
    if args.continuum_power:
        qso.flux = qso.cont

    # Resample real data onto lower resolution grid
    resamplingCondition = args.lowdv and args.lowdv > qso.dv
    if resamplingCondition:
        print("Resampling from %.2f to %.2f km/s" %(qso.dv, args.lowdv))
        qso = safeResample(qso, args.lowdv, args.keep_masked_pix)

    return qso

# This function is the main pipeline for reduction
# Pass mean_flux_hist=None to produce chunks
def pipeline(qso, f1, f2, meanFluxFunc, mean_flux_hist, args, disableChunk=False):
    qso = cleanup(qso, f1, f2, meanFluxFunc, args)

    # If computing mean flux end the pipeline here.
    if mean_flux_hist:
        mean_flux_hist.addSpectrum(qso, qso.s2n_lya**2/qso.dv, f1, f2, args.compute_scatter_error)
        return

    # If computing continuum power, approximate the error as the error on f.
    # cleanup swapped flux with continuum and resampled already
    if args.continuum_power:
        meanC     = np.mean(qso.flux)
        qso.flux /= meanC
        qso.error *= qso.flux
        qso.flux  -= 1
    else:
        qso.flux, qso.error = convert2DeltaFlux(qso.wave, qso.flux, qso.error, meanFluxFunc, args)

    # Skip short spectrum
    MAX_NO_PIXELS = int(fid.LIGHT_SPEED * np.log(fid.LYA_LAST_WVL/fid.LYA_FIRST_WVL) / qso.dv)
    isShort = lambda x: (args.skip and len(x) < MAX_NO_PIXELS * args.skip) or len(x)==0
    if isShort(qso.wave):
        raise ValueError("Short spectrum", len(qso.wave), MAX_NO_PIXELS)

    if not disableChunk and args.chunk_dyn:
        waves, fluxes, errors = so.chunkDynamic(qso.wave, qso.flux, qso.error, MAX_NO_PIXELS)
    elif not disableChunk and args.chunk_fixed:
        NUMBER_OF_CHUNKS = 3
        FIXED_CHUNK_EDGES = np.linspace(f1, f2, num=NUMBER_OF_CHUNKS+1)
        waves, fluxes, errors = so.divideIntoChunks(qso.wave, qso.flux, qso.error, \
            qso.z_qso, FIXED_CHUNK_EDGES)
    else:
        waves  = [qso.wave]
        fluxes = [qso.flux]
        errors = [qso.error]

    waves  = [x for x in waves  if not isShort(x)]
    fluxes = [x for x in fluxes if not isShort(x)]
    errors = [x for x in errors if not isShort(x)]
    
    if len(waves) == 0:
        raise ValueError("Empty chunks", len(waves))

    specres_list.add((qso.specres, qso.dv))
    return waves, fluxes, errors, qso.specres, qso.dv

# ------------------------
# Iterator functions
# ------------------------
def getFileIterator(dataset, directory):
    if dataset == 'KOD':
        set_iter = qio.KODIAQ_QSO_Iterator(directory, clean_pix=False)
    elif dataset == 'MOCK':
        set_iter = glob.glob(ospath_join(directory, "*.dat"))
    else:
        set_iter = glob.glob(ospath_join(directory, "*.fits"))

    return set_iter

def readFile(it, dataset, f1, f2, args):
    if dataset == 'KOD':
        obs_iter = qio.KODIAQ_OBS_Iterator(it)
        if args.coadd_kodiaq:
            # Co-add multiple observations
            qso = obs_iter.coaddObservations(args.coadd_kodiaq)
            s2n_this = qso.getS2NLya(f1, f2)
        else:
            # Pick highest S2N obs
            qso, s2n_this = obs_iter.maxLyaObservation(f1, f2)
        qso.qso_name = qso.qso_name.replace(" ", "")
        qso.print_details()

    elif dataset == 'XQ':
        qso = qio.XQ100Fits(it, correctSeeing=True)
        s2n_this = qso.getS2NLya(f1, f2)
        qso.qso_name = qso.qso_name.replace(" ", "")+"_"+qso.arm
        print(qso.qso_name)

    elif dataset == 'UVE':
        qso = qio.SQUADFits(it, correctSeeing=True, corrError=True)
        s2n_this = qso.getS2NLya(f1, f2)
        qso.qso_name = qso.qso_name.replace(" ", "")
        print(qso.qso_name, flush=True)

    elif dataset == "MOCK":
        qso = qio.BinaryQSO(it, 'r')
        s2n_this = qso.getS2NLya(f1, f2)
        qso.qso_name = qso.qso_name.replace(" ", "")
        print(qso.qso_name, flush=True)

    if s2n_this == -1:
        raise Exception("SKIP: No Lya or Side Band coverage!")

    if dataset == 'UVE' and qso.flag != '0':
        raise Exception("SKIP: Spec. status is not 0.")

    if s2n_this/np.sqrt(qso.dv) < args.sn_cut:
        raise Exception("SKIP: Does not pass S/N cut.")

    zmin = max(2.9, args.z_forest_min) if dataset=='XQ' else args.z_forest_min
    zmax = min(4.3, args.z_forest_max) if dataset=='XQ' else args.z_forest_max
    qso.cutForestAnalysisRegion(f1, f2, zmin, zmax)

    qso.s2n_lya = s2n_this
    return qso

def computeMeanFlux(directory, dataset, f1, f2, settings_txt, args):
    print("Calculating the mean flux....")
    mf_hist = so.MeanFluxHist(args.z_forest_min, args.z_forest_max)

    # Use a fiducial mean flux to ID DLAs.
    meanFluxFunc = fid.meanFluxFG08
        
    for it in getFileIterator(dataset, directory):
        try:
            qso = readFile(it, dataset, f1, f2, args)
            # Add Ly-a fluct as error here
            qso.addLyaFlucErrors()
            if args.mean_flux_lowdv:
                safeResample(qso, args.mean_flux_lowdv)
            pipeline(qso, f1, f2, meanFluxFunc, mf_hist, args)
        except Exception as e:
            print(e)
            continue
    
    mf_hist.getMeanStatistics(args.compute_scatter_error)
    mf_hist.saveHistograms(ospath_join(args.OutputDir, "%s-stats%s"%(dataset, settings_txt)))

    # Fit mean flux
    finite_indices = np.isfinite(mf_hist.mean_flux)

    fit_F = mf_hist.mean_flux[finite_indices]
    fit_e = mf_hist.mean_error2[finite_indices]
    fit_z = mf_hist.hist_redshifts[finite_indices]

    pnew, pcov = fid.fitBecker13MeanFlux(fit_z, fit_F, fit_e)
    print("Mean flux is fit!")

    return pnew

# This function is wrapper for iterations of each data set
def iterateSpectra(directory, dataset, f1, f2, meanFluxFunc, settings_txt, args):
    if args.real_data and args.side_band == 0:
        if args.compute_mean_flux:
            p_meanf = computeMeanFlux(directory, dataset, f1, f2, settings_txt, args)
            meanFluxFunc = lambda z: fid.evaluateBecker13MeanFlux(z, *p_meanf)
        elif dataset == 'KOD':
            meanFluxFunc = lambda z: fid.evaluateBecker13MeanFlux(z, *fid.KODIAQ_MFLUX_PARAMS)
        elif dataset == 'XQ':
            meanFluxFunc = lambda z: fid.evaluateBecker13MeanFlux(z, *fid.XQ100_MFLUX_PARAMS)
        elif dataset == 'UVE':
            meanFluxFunc = lambda z: fid.evaluateBecker13MeanFlux(z, *fid.UVES_MFLUX_PARAMS)
        # no need for else, meanFluxFunc is already set anyway

    elif args.real_data and args.side_band != 0:
        meanFluxFunc = lambda z: 1.0

    for it in getFileIterator(dataset, directory):
        print("********************************************", flush=True)
        try:
            qso = readFile(it, dataset, f1, f2, args)
        except Exception as e:
            print(e)
            continue

        try:
            wave, fluxes, errors, lspecr, pixw = pipeline(qso, f1, f2, meanFluxFunc, None, args)
        except ValueError as ve:
            print(ve.args)
            continue
        except Exception as e:
            print(e)
            continue
            
        nchunks = len(wave)

        temp_fname = ["%s-%s-%d_%dA_%dA%s.dat" % (dataset, qso.qso_name, nc, \
            wave[nc][0], wave[nc][-1], settings_txt) for nc in range(nchunks)]
        
        spectral_record_list.append(dataset, qso.qso_name, qso.s2n_lya/np.sqrt(qso.dv), \
            qso.coord, temp_fname)

        filename_list.extend(temp_fname) 

        if not args.nosave:
            saveData(wave, fluxes, errors, temp_fname, qso, lspecr, pixw, args)

if __name__ == '__main__':
    # Arguments passed to run the script
    print("Parsing arguments....", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("OutputDir", help="Output directory")
    parser.add_argument("--seed", help="Seed to generate random numbers.", type=int, default=68970)

    parser.add_argument("--KODIAQDir", help="Directory of KODIAQ")
    parser.add_argument("--coadd-kodiaq", type=float, \
        help="Co-adds different observations of the same quasar onto given dv grid.")

    parser.add_argument("--XQ100Dir", help="Directory of XQ100")
    parser.add_argument("--UVESSQUADDir", help="Directory of SQUAD/UVES")
    parser.add_argument("--MockDir", help="Directory for full flux mocks to READ.")

    parser.add_argument("--separation", type=float, default=2.5, \
        help="Maximum separation in arc sec. Default: %(default)s")
    
    parser.add_argument("--sn-cut", type=float, default=-1, \
        help="S/N cut per sqrt of km/s.")

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
        help="Use this constant error for mocks when passed instead. " \
        "Otherwise data errors are used.")

    parser.add_argument("--gauss", help="Generate Gaussian mocks", action="store_true")
    parser.add_argument("--without_z_evo", help="Turn of redshift evolution", action="store_true")
    parser.add_argument("--lowdv", help="Resamples grid to this pixel size (km/s) when passed", \
        type=float)
    parser.add_argument("--find-dlas", action="store_true", \
        help="Use simple DLA finder to find DLAs, then save to a file.")
    parser.add_argument("--mask-dlas", action="store_true")
    parser.add_argument("--add-dlas-to-mocks", action="store_true")
    
    parser.add_argument("--mask-sigma-percentile", help="Mask outliers by percentile by sigma.", \
        type=float)
    parser.add_argument("--mask-spikes-zscore", help="Mask spikes by given zscore.", type=float)
    parser.add_argument("--keep-masked-pix", help="Assign large errors.", action="store_true")
    parser.add_argument("--z-forest-min", help="Lower end of the forest. Default: %(default)s", \
        type=float, default=1.7)
    parser.add_argument("--z-forest-max", help="Upper end of the forest. Default: %(default)s", \
        type=float, default=4.3)
    
    parser.add_argument("--side-band", type=int, default=0, help="Side band. Default: %(default)s")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--continuum-power", action="store_true", \
        help="Use continuum instead of flux. Compute dC/C-bar, C-bar is the full forest average.")
   
    parser.add_argument("--compute-mean-flux", action="store_true")
    parser.add_argument("--mean-flux-lowdv", type=float, \
        help="Resample the grid using inverse variance without LSS fluctuations for mean flux" \
        " calculation. 300 is recommended.")
    parser.add_argument("--compute-scatter-error", action="store_true")

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

    # Pick mean flux function for the mocks
    if args.gauss:
        meanFluxFunc = fid.meanFluxFG08
    else:
        meanFluxFunc = lm.lognMeanFluxGH

    # Decide if it's real data
    if args.real_data:
        settings_txt  = ''
    else:
        settings_txt  = '_gaussian' if args.gauss else '_lognormal' 

    # Set settings text
    settings_txt += '_dv%.1f' % args.lowdv if args.lowdv else ''
    settings_txt += '_noz' if args.without_z_evo else ''
    settings_txt += '_sb%d'%args.side_band if args.side_band else ''
    settings_txt += '_masked-dlas' if args.mask_dlas else ''

    txt_basefilename  = "%s/highres%s" % (args.OutputDir, settings_txt)

    saveParameters(txt_basefilename, forest_1, forest_2, args)
    # if args.find_dlas:
    #     dla_file = open(txt_basefilename+"_dlas.txt", 'w')
    #     dla_file.write("z_dlas,nhi_dlas,qso_name,set,ra,dec\n")
    # ------------------------------

    if not args.real_data:
        lya_m = lm.LyaMocks(args.seed, N_CELLS=args.ngrid, DV_KMS=args.griddv, \
            REDSHIFT_ON=not args.without_z_evo, GAUSSIAN_MOCKS=args.gauss)

    # ------------------------------    
    # Start with KODIAQ
    if args.KODIAQDir and not args.continuum_power:
        print("RUNNING ON KODIAQ.........", flush=True)        
        # Start iterating quasars in KODIAQ sample
        # Each quasar has multiple observations
        # Pick the one with highest signal to noise in Ly-alpha region
        iterateSpectra(args.KODIAQDir, 'KOD', forest_1, forest_2, meanFluxFunc, settings_txt, args)
    # ------------------------------
    # XQ-100
    if args.XQ100Dir:
        print("RUNNING ON XQ-100.........", flush=True)
        iterateSpectra(args.XQ100Dir, 'XQ', forest_1, forest_2, meanFluxFunc, settings_txt, args)
    # ------------------------------
    # UVES/SQUAD
    if args.UVESSQUADDir:
        print("RUNNING ON SQUAD/UVES.........", flush=True)
        iterateSpectra(args.UVESSQUADDir, 'UVE', forest_1, forest_2, meanFluxFunc, settings_txt, args)
    # ------------------------------
    # MOCKS
    if args.MockDir:
        print("RUNNING ON MOCKS.........", flush=True)
        iterateSpectra(args.MockDir, 'MOCK', forest_1, forest_2, meanFluxFunc, settings_txt, args)
    # ------------------------------

    # if args.find_dlas:
    #     dla_file.close()

    temp_fname = ospath_join(args.OutputDir, "specres_list.txt")
    print("Saving spectral resolution values as ", temp_fname)
    qio.saveListByLine(specres_list, temp_fname)

    # Save the list of files in a txt
    temp_fname = ospath_join(args.OutputDir, "file_list_qso.txt")
    print("Saving chunk spectra file list as ", temp_fname)
    qio.saveListByLine(filename_list, temp_fname)

    spectral_record_list.saveAsTable(ospath_join(args.OutputDir, "spr-all.csv"))
    nondups = spectral_record_list.getNonDuplicates(args.separation)
    nondups.saveAsTable(ospath_join(args.OutputDir, "spr-nonduplicates.csv"))

    filename_list = []
    for fl in map(lambda x: x.fnames, nondups.spr):
        filename_list.extend(fl)
    temp_fname = ospath_join(args.OutputDir, "file_list_qso-nonduplicates.txt")
    print("Saving chunk spectra file list as ", temp_fname)
    qio.saveListByLine(filename_list, temp_fname)



























