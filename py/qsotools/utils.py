import logging
import time
from itertools import groupby

import numpy as np

def decomposePiccaFname(picca_fname):
    i1 = picca_fname.rfind('[')+1
    i2 = picca_fname.rfind(']')

    basefname = picca_fname[:i1-1]
    hdunum = int(picca_fname[i1:i2])

    return (basefname, hdunum)

def getPiccaFList(fnames_spectra):
    decomp_list = [decomposePiccaFname(fl.rstrip()) for fl in fnames_spectra]
    decomp_list.sort(key=lambda x: x[0])

    new_fnames = []
    for base, hdus in groupby(decomp_list, lambda x: x[0]):
        new_fnames.append((base, list(map(lambda x: x[1], hdus))))

    fnames_spectra = new_fnames

class Progress(object):
    """Utility class to log progress. Initialize with total number of operations."""
    def __init__(self, total, percThres=5):
        self.i = 0
        self.total = total
        self.percThres = percThres
        self.last_progress = 0
        self.start_time = time.time()

    def increase(self):
        self.i+=1
        curr_progress = int(100*self.i/self.total)
        print_condition = (curr_progress-self.last_progress >= self.percThres) or (self.i == 0)

        if print_condition:
            etime = (time.time()-self.start_time)/60 # min
            logging.info(f"Progress: {curr_progress}%. Elapsed time {etime:.1f} mins.")
            self.last_progress = curr_progress

class SubsampleCov(object):
    """docstring for SubsampleCov"""
    def __init__(self, nbins, nsamples, is_weighted=False):
        self.nbins = nbins
        self.nsamples = nsamples
        self.isample = 0
        self.is_weighted = is_weighted
        self.is_normalized = False

        self.all_measurements = np.zeros((nsamples, nbins))
        if self.is_weighted:
            self.all_weights = np.zeros((nsamples, nbins))

    # The measurement provided (xvec) should already be weighted and
    # not normalized if wvec passed
    def addMeasurement(self, xvec, wvec=None):
        if (wvec is None) and self.is_weighted:
            raise RuntimeError("SubsampleCov requires weights")
        if (wvec is not None) and (not self.is_weighted):
            raise RuntimeError("SubsampleCov unexpected weights")

        self.all_measurements[self.isample] += xvec

        if (wvec is not None) and self.is_weighted:
            self.all_weights[self.isample] += wvec

        self.isample = (self.isample+1)%self.nsamples

    def _normalize(self):
        if self.is_weighted:
            self.all_measurements /= self.all_weights + np.finfo(float).eps
            self.all_weights /= np.sum(self.all_weights, axis=0) + np.finfo(float).eps
        else:
            self.all_weights = np.ones(self.nsamples)/self.nsamples

        self.is_normalized = True

    def getMean(self):
        if not self.is_normalized:
            self._normalize()

        mean_xvec = np.sum(self.all_measurements*self.all_weights, axis=0)

        return mean_xvec

    # This is a jackknife estimate of the variance
    def getMeanNCov(self, bias_correct=False):
        if not self.is_normalized:
            self._normalize()

        mean_xvec = self.getMean()

        # remove one measurement, then renormalize
        jack_i = (mean_xvec - self.all_measurements*self.all_weights)/(1-self.all_weights)
        mean_jack = np.mean(jack_i, axis=0)

        if bias_correct:
            bias_jack = (self.nsamples-1) * (mean_jack-mean_xvec)
            mean_xvec -= bias_jack

        xdiff = jack_i - mean_jack

        cov = np.dot(xdiff.T, xdiff)*(self.nsamples-1)/self.nsamples

        return mean_xvec, cov




















