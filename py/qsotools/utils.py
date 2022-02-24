import logging
import time

import numpy as np

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

        self.all_measurements = np.zeros((nsamples, nbins))
        if self.is_weighted:
            self.all_weights = np.zeros((nsamples, nbins))

    # The measurement provided (xvec) should not be normalized
    def addMeasurement(self, xvec, wvec=None):
        if (wvec is None) and self.is_weighted:
            raise RuntimeError("SubsampleCov requires weights")
        if (wvec is not None) and (not self.is_weighted):
            raise RuntimeError("SubsampleCov unexpected weights")

        self.all_measurements[self.isample] += xvec

        if (wvec is not None) and self.is_weighted:
            self.all_weights[self.isample] += wvec

        self.isample = (self.isample+1)%self.nsamples

    def getMean(self):
        aweights = self.all_weights if self.is_weighted else 1./self.nsamples

        mean_xvec = np.sum(self.all_measurements, axis=0)

        if self.is_weighted:
            norm = np.sum(aweights, axis=0)
            norm[norm <= 0] = 1
        else:
            norm = 1

        mean_xvec /= norm

        return mean_xvec

    def getMeanNCov(self):
        aweights = self.all_weights if self.is_weighted else 1./self.nsamples

        mean_xvec = self.getMean()

        weighted_xdiff = self.all_measurements - aweights * mean_xvec
        cov = weighted_xdiff.T.dot(weighted_xdiff)

        if self.is_weighted:
            norm = np.sum(aweights, axis=0)
            norm[norm <= 0] = 1
            norm2 = norm * norm[:, None]
        else:
            norm2 = 1

        cov /= norm2

        return mean_xvec, cov



















