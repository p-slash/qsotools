import numpy as np
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d as scipy_interp1d

def getCrossSpectrum(f1_v, f2_v, dv, k_edges):
    k = 2*np.pi*np.fft.rfftfreq(f1_v.size, dv)
    L = f1_v.size * dv

    f1_k = np.fft.rfft(f1_v) * dv
    f2_k = np.fft.rfft(f2_v) * dv
    cross_power = np.abs(np.conj(f1_k) * f2_k) / L

    binned_power,  _, binnumber = binned_statistic(k, cross_power, statistic='sum', bins=k_edges)
    counts = np.bincount(binnumber, minlength=len(k_edges)+1)

    return binned_power, counts

class Reconstructor():
    """Works only in a single redshift"""
    def __init__(self, k_edges=np.arange(30)*0.005):
        self.k_edges = k_edges
        self.k_bins  = (k_edges[1:] + k_edges[:-1]) / 2
        self.cross = np.zeros(k_edges.size-1)
        self.auto = np.zeros(k_edges.size-1)
        self.wiener = np.zeros(k_edges.size-1)

    def addSpectrum(self, delta_v, dv, delta_v_true=None):
        if delta_v_true is None:
            delta_v_true = delta_v

        delta_1 = delta_v**2
        
        self.cross += getCrossSpectrum(delta_v_true, delta_1, dv, self.k_edges)[0]
        self.auto  += getCrossSpectrum(delta_1, delta_1, dv, self.k_edges)[0]

    def calcWiener(self):
        self.wiener = self.cross / self.auto
        self.wiener_interp = scipy_interp1d(self.k_bins, self.wiener, bounds_error=False, \
            fill_value=0)
    
    def getReconstructedField(self, delta_v, dv):
        k = 2*np.pi*np.fft.rfftfreq(delta_v.size, dv)

        delta_1 = delta_v**2
        delta_rec_k = self.wiener_interp(k) * np.fft.rfft(delta_1) * dv

        return np.fft.irfft(delta_rec_k, delta_v.size) / dv




