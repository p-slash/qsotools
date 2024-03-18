from pkg_resources import resource_filename

import numpy as np
import matplotlib.pyplot as plt

from qsotools.fiducial import LYA_WAVELENGTH

import qsonic.io
import qsonic.catalog
from qsonic.mathtools import fft_gaussian_smooth

plt.style.use(resource_filename('qsotools', 'alluse.mplstyle'))

EMISSION_LINES = {
    r'Ly$\beta$': (1025.72, 40),
    r'Ly$\alpha$': (1215.67, 90),
    r"N V": (1240.81, 59),
    'Si IV': (1393.76, 30),
    'C IV': (1548.202, 42),
    'Mg II': (2800, 42)
}

REGIONS = {
    r"Ly$\alpha$ forest": (1050., 1180.),
    r"SB 1": (1268., 1380.),
    r"SB 2": (1409., 1523.)
}


def getMaxFlux(specobj, wave_c, dwave):
    ymax = -1
    for arm, wave in specobj.wave.items():
        w = np.abs(wave_c - wave) < dwave
        ymax = max(ymax, specobj.flux[arm][w].max())

    return ymax


def getMedianFlux(specobj, wave_max):
    wave = np.concatenate(
        [_ for _ in specobj.wave.values()])
    flux = np.concatenate(
        [_ for _ in specobj.flux.values()])
    w = wave < wave_max
    median = np.median(flux[w])
    mad = np.median(np.abs(flux[w] - median))
    return median, mad


class DesiPlotter():
    def __init__(self, cat_fname, spectrumdir, is_mock, is_tile):
        self.catalog = qsonic.catalog.read_quasar_catalog(
            cat_fname, is_mock=is_mock, is_tile=is_tile)
        self.spectrumdir = spectrumdir
        self.is_mock = is_mock

        self.readerFunction = qsonic.io.get_spectra_reader_function(
            spectrumdir, ['B', 'R', 'Z'], is_mock, skip_resomat=True,
            read_true_continuum=is_mock, is_tile=is_tile)

    def plot_spectrum_v3(
            self, targetid, coadd=False, smoothing_kernel=0, shift_z=0,
            emlines=EMISSION_LINES, regions=REGIONS,
            plot_ivar=False, figsize=(13, 5)
    ):
        idx = np.nonzero(targetid == self.catalog['TARGETID'])[0][0]
        cat1 = self.catalog[idx]
        program = cat1['PROGRAM']
        survey = cat1['SURVEY']
        zqso = cat1['Z'] + shift_z

        specobj = self.readerFunction(cat1)
        if plot_ivar:
            specobj._flux = specobj.ivar
        if coadd:
            specobj.simple_coadd()

        if smoothing_kernel > 0:
            for arm, f in specobj.flux.items():
                specobj.flux[arm] = fft_gaussian_smooth(f, smoothing_kernel)

        # Start plotting
        plt.figure(figsize=figsize)

        ymax = getMaxFlux(specobj, (1 + zqso) * LYA_WAVELENGTH, 50.)

        for key, value in emlines.items():
            wave_c = value[0] * (1 + zqso)
            y = getMaxFlux(specobj, wave_c, 20.) * 1.03
            ys = [y, y + ymax / 12]
            plt.plot([wave_c, wave_c], ys, c="xkcd:violet", lw=2.5, alpha=0.8)
            plt.text(
                wave_c, y + 1.3 * ymax / 12, key,
                c="xkcd:violet",
                horizontalalignment='center',
                fontsize=16)
            del wave_c

        jj = 0
        for key, value in regions.items():
            w1 = value[0] * (1 + zqso)
            w2 = value[1] * (1 + zqso)
            plt.axvspan(w1, w2, fc=plt.cm.tab10(jj), alpha=0.1)
            plt.text(
                (w1 + w2) / 2, ymax * 1.1, key,
                c=plt.cm.tab10(jj), fontweight='bold',
                horizontalalignment='center', fontsize=16)
            jj += 1

        for arm, wave in specobj.wave.items():
            plt.plot(wave, specobj.flux[arm], alpha=1, lw=0.7)

        median, mad = getMedianFlux(specobj, (1 + zqso) * 1300.)
        plt.ylim(median - 6 * mad, ymax * (1.25))

        if plot_ivar:
            plt.ylabel("IVAR")
        else:
            plt.ylabel(r"Flux [$10^{-17}$ erg$~/~$(cm$^2~$s$~\AA$)]")
        plt.xlabel(r"Observed wavelength [$\AA$]")
        # qplt.ticks_makeup(plt.gca())

        # def ticks_makeup(ax):
        ax = plt.gca()
        ax.tick_params(
            direction='in', which='major', length=7, width=1,
            right=True, top=False)
        ax.tick_params(
            direction='in', which='minor', length=4, width=1,
            right=True, top=False)

        secx = ax.secondary_xaxis(
            'top', functions=(lambda x: x / (1 + zqso), lambda x: x / (1 + zqso))
        )
        secx.tick_params(
            direction='in', which='major', length=7, width=1,
            left=False, right=False, top=True)
        secx.tick_params(
            direction='in', which='minor', length=4, width=1,
            left=False, right=False, top=True)
        plt.setp(secx.get_xticklabels(), fontsize=14)
        secx.set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=16)

        plotfname = f"spectrum-z{zqso:.2f}-{targetid}-{survey}-{program}-v3.1.png"
        print(plotfname)

        return ax, secx
