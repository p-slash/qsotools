from pkg_resources import resource_filename

import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt


class P1DMeasurements():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    def load_walther(self, load=True):
        if not load:
            return

        fname = resource_filename(
            'qsotools', 'p1d-measurements/walther_metalmasked_power_table5.tsv'
        )
        mw_ps_metal_masked = ascii.read(fname, data_start=3)
        # mw_nz = 9
        # mw_nk = 22

        z = np.array(mw_ps_metal_masked['z'], dtype=np.double)
        z = np.unique(np.round(z, decimals=1))
        nz = z.size
        nk = len(mw_ps_metal_masked) // nz

        k = np.array(
            mw_ps_metal_masked['k'], dtype=np.double
        ).reshape((nz, nk))
        kppi = np.array(
            mw_ps_metal_masked['k.P(k)/pi'], dtype=np.double
        ).reshape((nz, nk))
        ekpi = np.array(
            mw_ps_metal_masked['e_k.P(k)/pi'], dtype=np.double
        ).reshape((nz, nk))

        self.measurements['Walther+2017'] = {
            'z': z,
            'k': k,
            'kppi': kppi,
            'ekpi': ekpi,
            'alpha': 0.4,
            'color': P1DMeasurements.colors[-2]
        }

    def load_eboss(self, load=True):
        if not load:
            return

        fname_data = resource_filename(
            'qsotools', 'p1d-measurements/chabanier_Pk1D_data.txt'
        )
        fname_syst = resource_filename(
            'qsotools', 'p1d-measurements/chabanier_Pk1D_syst.txt'
        )
        # z=[2.2, 4.6]
        chab_table = ascii.read(fname_data)
        chab_syst_table = np.loadtxt(fname_syst)
        # chab19_nk = 35
        # chab19_nz = 13

        z = np.array(chab_table['zLya'], dtype=np.double)
        z = np.unique(np.round(z, decimals=1))
        nz = z.size
        nk = len(chab_table) // nz

        k = np.array(chab_table['k'], dtype=np.double).reshape((nz, nk))
        p = np.array(chab_table['PLya'], dtype=np.double).reshape((nz, nk))
        e = np.array(chab_table['stat'], dtype=np.double)
        e = np.sqrt(
            np.sum(chab_syst_table**2, axis=1) + e**2
        ).reshape((nz, nk))

        self.measurements['Chabanier et al. (2019)'] = {
            'z': z,
            'k': k,
            'kppi': k * p / np.pi,
            'ekpi': k * e / np.pi,
            'alpha': 0.5,
            'color': P1DMeasurements.colors[-3]
        }

    def load_karacayli(self, load=True):
        if not load:
            return

        fname = resource_filename(
            'qsotools',
            'p1d-measurements/final-conservative-p1d-karacayli_etal2021.txt'
        )
        karac_tbl = ascii.read(fname, format='fixed_width')

        z = np.array(karac_tbl['z'], dtype=np.double)
        z = np.unique(np.round(z, decimals=1))
        nz = z.size
        nk = len(karac_tbl) // nz

        # karac_nz = 14
        # karac_nk = 13

        k = np.array(karac_tbl['k'], dtype=np.double).reshape((nz, nk))
        p = np.array(karac_tbl['P'], dtype=np.double).reshape((nz, nk))
        e = np.array(karac_tbl['e'], dtype=np.double).reshape((nz, nk))

        self.measurements['Karaçaylı et al. (2022)'] = {
            'z': z,
            'k': k,
            'kppi': k * p / np.pi,
            'ekpi': k * e / np.pi,
            'alpha': 0.5,
            'color': P1DMeasurements.colors[-4]
        }

    def load_karacayli_desi_edrp(self, load=True):
        if not load:
            return

        fname = resource_filename(
            'qsotools',
            'p1d-measurements/karacayli23_desi-edrp-lyasb1subt'
            '-p1d-summary-results.txt')

        karac_tbl = np.array(ascii.read(fname))

        z = np.array(karac_tbl['z'], dtype=np.double)
        zbins = np.unique(np.round(z, decimals=1))

        karac_tbl['p_final'] *= karac_tbl['kc'] / np.pi
        karac_tbl['e_total'] *= karac_tbl['kc'] / np.pi

        wzlist = [np.isclose(z, zb) for zb in zbins]
        k = [karac_tbl['kc'][wz] for wz in wzlist]
        kppi = [karac_tbl['p_final'][wz] for wz in wzlist]
        ekpi = [karac_tbl['e_total'][wz] for wz in wzlist]

        self.measurements['Karaçaylı et al. (2024)'] = {
            'z': zbins,
            'k': k,
            'kppi': kppi,
            'ekpi': ekpi,
            'alpha': 0.5,
            'color': P1DMeasurements.colors[-1]
        }

    def __init__(
            self, karacayli=False, eboss=False, walther=False,
            karacayli_desi_edrp=False
    ):
        self.measurements = {}
        self.load_eboss(eboss)
        self.load_walther(walther)
        self.load_karacayli(karacayli)
        self.load_karacayli_desi_edrp(karacayli_desi_edrp)

    def __bool__(self):
        return bool(self.measurements)

    def plot_one(self, zplot, ax, key):
        if key not in self.measurements:
            return

        data = self.measurements[key]
        wz = data['z'] == zplot
        if not any(wz):
            return

        iz = np.nonzero(wz)[0][0]

        plotted = ax.fill_between(
            data['k'][iz],
            data['kppi'][iz] - data['ekpi'][iz],
            data['kppi'][iz] + data['ekpi'][iz],
            label=key, alpha=data['alpha'], color=data['color'])

        return plotted

    def plot_all(self, zplot, ax=plt):
        fs = []

        for key in self.measurements.keys():
            fs.append(self.plot_one(zplot, ax, key))

        return fs
