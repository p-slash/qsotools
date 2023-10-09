from pkg_resources import resource_filename

import numpy as np
import matplotlib.pyplot as plt
import fitsio

plt.style.use(resource_filename('qsotools', 'alluse.mplstyle'))


def add_minor_grid(ax):
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=1)


class AttrFile():
    @staticmethod
    def variance_function(var_pipe, var_lss, eta):
        return eta * var_pipe + var_lss

    def __init__(self, fname, name):
        self.fname = fname
        self.name = name
        fattr = fitsio.FITS(fname)

        self.cont = fattr['CONT'].read()
        self.stacked_flux = fattr['STACKED_FLUX'].read()
        self.varfunc = fattr['VAR_FUNC'].read()

        hdr = dict(fattr['VAR_STATS'].read_header())
        varstats = fattr['VAR_STATS'].read().reshape(
            hdr['NWBINS'], hdr['NVARBINS'])
        hdr['data'] = varstats
        self.varstats = hdr

        fattr.close()

    def plot_varpipe_varobs(self, iwave, ax=None, plotfit=True, show=True):
        if ax is None:
            ax = plt.gca()

        data = self.varstats['data'][iwave]
        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        ax.errorbar(
            data['var_pipe'][w2], data['var_delta'][w2],
            yerr=data['e_var_delta'][w2],
            fmt='.', alpha=1, label="Nominal range")

        # Plot extended range
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100) & (~w2)
        ax.errorbar(
            data['var_pipe'][w2], data['var_delta'][w2],
            yerr=data['e_var_delta'][w2],
            fmt='.', alpha=1, label="Extended range")

        # Plot var_lss, eta fitting function
        if plotfit:
            w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100)
            var_pipe = data['var_pipe'][w2]
            var_pipe = var_pipe[var_pipe > 0]
            var_pipe = np.logspace(
                np.log10(var_pipe.min()), np.log10(var_pipe.max()),
                100)

            var_lss = self.varfunc['var_lss'][iwave]
            e_var_lss = self.varfunc['e_var_lss'][iwave]
            eta = self.varfunc['eta'][iwave]
            e_eta = self.varfunc['e_eta'][iwave]

            yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
            ymin = AttrFile.variance_function(
                var_pipe, var_lss - e_var_lss, eta - e_eta)
            ymax = AttrFile.variance_function(
                var_pipe, var_lss + e_var_lss, eta + e_eta)

            ax.plot(var_pipe, yfit, 'k-', label="Fit")
            ax.fill_between(var_pipe, ymin, ymax, alpha=0.6, fc='grey')

        # Add texts
        meanl = np.mean(data['wave'])
        ax.text(
            0.06, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.5, 'boxstyle': 'round', "lw": 2}
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Observed variance")

        add_minor_grid(ax)
        ax.legend(loc="lower right")

        if show:
            plt.show()

    def plot_varpipe_meandelta(self, iwave, ax=None, show=True):
        if ax is None:
            ax = plt.gca()

        data = self.varstats['data'][iwave]
        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        ax.errorbar(
            data['var_pipe'][w2], data['mean_delta'][w2],
            yerr=np.sqrt(data['var_delta'][w2] / data['num_pixels'][w2]),
            fmt='.', alpha=1, label="Nominal range")

        # Plot extended range
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100) & (~w2)
        ax.errorbar(
            data['var_pipe'][w2], data['mean_delta'][w2],
            yerr=np.sqrt(data['var_delta'][w2] / data['num_pixels'][w2]),
            fmt='.', alpha=1, label="Extended range")

        # Add texts
        meanl = np.mean(data['wave'])
        ax.text(
            0.06, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.5, 'boxstyle': 'round', "lw": 2}
        )

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Mean delta")
        ax.axhline(0, c='k')

        add_minor_grid(ax)
        ax.legend(loc="lower right")

        if show:
            plt.show()

    def plot_varpipe_varobs_ratio(
            self, iwave, ax=None, show=True, print_chi2=True, ylim=(0.9, 1.2)
    ):
        if ax is None:
            ax = plt.gca()

        var_lss = self.varfunc['var_lss'][iwave]
        e_var_lss = self.varfunc['e_var_lss'][iwave]
        eta = self.varfunc['eta'][iwave]
        e_eta = self.varfunc['e_eta'][iwave]
        data = self.varstats['data'][iwave]

        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        var_pipe = data['var_pipe'][w2]
        var_obs = data['var_delta'][w2]
        e_var_obs = data['e_var_delta'][w2]
        yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
        ax.errorbar(
            var_pipe, var_obs / yfit, yerr=e_var_obs / yfit,
            fmt='.', alpha=1, label="Nominal range")

        if print_chi2:
            d = var_obs - yfit
            chi2 = np.sum((var_obs - yfit)**2 / e_var_obs**2)
            print(f"Diagonal chi2 / dof: {chi2:.2f} / {d.size}")
            invcov = np.linalg.inv(data['cov_var_delta'][w2, :][:, w2])
            chi2 = np.dot(d, invcov.dot(d))
            print(f"Invcov chi2 / dof: {chi2:.2f} / ({d.size} - 2)")

        # Plot extended range
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100) & (~w2)
        var_pipe = data['var_pipe'][w2]
        yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
        ax.errorbar(
            var_pipe, data['var_delta'][w2] / yfit,
            yerr=data['e_var_delta'][w2] / yfit,
            fmt='.', alpha=1, label="Extended range")

        # Plot shaded tilt
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100)
        var_pipe = data['var_pipe'][w2]
        var_pipe = var_pipe[var_pipe > 0]
        var_pipe = np.logspace(
            np.log10(var_pipe.min()), np.log10(var_pipe.max()),
            100)

        yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
        ymin = AttrFile.variance_function(
            var_pipe, var_lss - e_var_lss, eta - e_eta) / yfit
        ymax = AttrFile.variance_function(
            var_pipe, var_lss + e_var_lss, eta + e_eta) / yfit
        ax.fill_between(var_pipe, ymin, ymax, alpha=0.6, fc='grey')

        # Add texts
        meanl = np.mean(data['wave'])
        ax.text(
            0.76, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.5, 'boxstyle': 'round', "lw": 2}
        )

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Observed variance / Model")
        ax.axhline(1, c='k')

        add_minor_grid(ax)
        ax.legend(loc="lower right", ncol=2)

        if isinstance(ylim, tuple):
            ax.set_ylim(*ylim)

        if show:
            plt.show()

    def plot_one_cov(self, iwave, cmap='RdBu_r', show=True):
        data = self.varstats['data'][iwave]
        vp = data['var_pipe']
        w0 = vp > 0

        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        if w2.sum() < 5:
            print("Used extended cuts due to low statistics")
            w2 = ((data['num_qso'] >= self.varstats['MINNQSO'] // 2)
                  & (data['num_pixels'] >= self.varstats['MINNPIX'] // 2))

        w2 = w2[w0]
        vp = vp[w0]
        a10 = 10**(np.log10(vp[1]) - np.log10(vp[0]))
        vp_edges = 2 * vp / (1 + a10)
        vp_edges = np.append(vp_edges, vp_edges[-1] * a10)

        e = np.sqrt(data['cov_var_delta'].diagonal())[w0]
        e = np.where(e == 0, 1, e)

        matrix = data['cov_var_delta'][w0, :][:, w0] / np.outer(e, e)
        matrix[np.isnan(matrix)] = 0

        fig, ax = plt.subplots()

        im = ax.pcolormesh(
            vp_edges, vp_edges, matrix,
            cmap=cmap, shading='flat', vmin=-1, vmax=1)

        m2 = np.zeros_like(matrix) * np.nan
        m2[~w2, :] = 1
        m2[:, ~w2] = 1
        plt.pcolormesh(
            vp_edges, vp_edges, m2, shading='flat',
            alpha=0.2, cmap='Greys_r')

        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xticks(np.logspace(-4, 1, 6))
        ax.set_yticks(np.logspace(-4, 1, 6))
        ax.set_xlim(vp[vp > 0].min(), vp.max())
        ax.set_ylim(vp[vp > 0].min(), vp.max())
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Pipeline variance")

        cbar = fig.colorbar(im, ticks=np.linspace(-1, 1, 6))
        # cbar.set_label(cbarlbl, fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        meanl = np.mean(data['wave'])
        ax.text(
            0.06, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.9, 'boxstyle': 'round', "lw": 2}
        )

        if show:
            plt.show()

    @staticmethod
    def _parse_otherattrs(otherAttrs):
        if otherAttrs is None:
            otherAttrs = []
        elif isinstance(otherAttrs, AttrFile):
            otherAttrs = [otherAttrs]
        elif not isinstance(otherAttrs, list):
            raise Exception("otherAttrs must be a list or AttrFile.")
        return otherAttrs

    def plot_mean_cont(self, otherAttrs=None, ax=None, show=True):
        """
        Args:
            otherAttrs (list(AttrFile))
        """
        if ax is None:
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        for f in fattr:
            data = f.cont
            ax.errorbar(
                data['lambda_rf'], data['mean_cont'], data['e_mean_cont'],
                label=f.name, capsize=0)

        ax.set_ylabel("Mean continuum")
        ax.set_xlabel("Rest-frame wavelength [A]")
        add_minor_grid(ax)
        ax.legend()
        if show:
            plt.show()

    def plot_eta(self, otherAttrs=None, ax=None, show=True):
        if ax is None:
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        for f in fattr:
            varfunc = f.varfunc
            ax.errorbar(
                varfunc['lambda'], varfunc['eta'] - 1, varfunc['e_eta'],
                fmt='.-', label=f.name)

        add_minor_grid(ax)
        ax.set_ylabel(r"$\eta - 1$")
        ax.axhline(0, c='k')
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylim(-0.03, 0.02)
        plt.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.legend()

        if show:
            plt.show()

    def plot_varlss(self, otherAttrs=None, ax=None, show=True):
        if ax is None:
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        for f in fattr:
            varfunc = f.varfunc
            ax.errorbar(
                varfunc['lambda'], varfunc['var_lss'], varfunc['e_var_lss'],
                fmt='.-', label=f.name)

        add_minor_grid(ax)
        ax.set_ylabel(r"$\sigma^2_\mathrm{LSS}$")
        ax.set_xlabel("Wavelength [A]")
        ax.legend()

        if show:
            plt.show()

    def plot_stacked_flux(self, otherAttrs=None, ax=None, show=True):
        if ax is None:
            plt.figure(figsize=(12, 5))
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        alpha = max(0.5, min(1, 1.6 / len(fattr)))
        minwave = 100000
        maxwave = -1

        for f in fattr:
            data = f.stacked_flux
            minwave = min(minwave, data['lambda'].min())
            maxwave = max(maxwave, data['lambda'].max())
            ax.plot(
                data['lambda'], data['stacked_flux'] - 1,
                '-', label=f.name, alpha=alpha)

        ax.set_ylabel("Stacked flux - 1")
        ax.axhline(0, c='k')
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylim(-0.05, 0.05)

        xtickmin = int(np.round(minwave / 400) * 400)
        xtickmax = int(np.round(maxwave / 400) * 400)
        nticks = (xtickmax - xtickmin) // 400
        ax.set_xticks(np.linspace(xtickmin, xtickmax, nticks + 1))

        add_minor_grid(ax)

        plt.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.legend(ncol=2)

        if show:
            plt.show()

    def plot_all_varpipe_varobs(self):
        for i in range(self.varstats['NWBINS']):
            self.plot_varpipe_varobs(i)

    def plot_all_varpipe_varobs_ratio(self):
        for i in range(self.varstats['NWBINS']):
            self.plot_varpipe_varobs_ratio(i)

    def plot_all_covariances(self):
        for i in range(self.varstats['NWBINS']):
            self.plot_one_cov(i)

    def plot_all_varpipe_meandelta(self, inone=False, show=True):
        if not inone:
            for i in range(self.varstats['NWBINS']):
                self.plot_varpipe_meandelta(i)
            return

        ax = plt.gca()
        for iwave in range(self.varstats['NWBINS']):
            data = self.varstats['data'][iwave]
            meanl = np.mean(data['wave'])

            w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100)
            ax.errorbar(
                data['var_pipe'][w2], data['mean_delta'][w2],
                yerr=np.sqrt(data['var_delta'][w2] / data['num_pixels'][w2]),
                fmt='.', alpha=1, label=f"{meanl:.0f}")

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Mean delta")
        ax.axhline(0, c='k')

        add_minor_grid(ax)
        ax.legend(loc="lower right")

        if show:
            plt.show()
