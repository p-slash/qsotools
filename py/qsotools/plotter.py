import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.interpolate import RectBivariateSpline
from astropy.io import ascii

TICK_LBL_FONT_SIZE = 18
AXIS_LBL_FONT_SIZE = 20

def set_topax_makeup(top_ax, majorgrid=True, ymin=1e-4, ymax=0.5):
    top_ax.grid(majorgrid, which = 'major')
    top_ax.set_yscale("log")
    top_ax.set_xscale("log")
    top_ax.set_ylim(ymin=1e-4, ymax=0.5)

    top_ax.tick_params(which='major', direction='in', length=7, width=1)
    top_ax.tick_params(which='minor', direction='in', length=4, width=0.8)

    plt.setp(top_ax.get_yticklabels(), fontsize = TICK_LBL_FONT_SIZE)
    top_ax.set_ylabel(r'$kP/\pi$', fontsize = AXIS_LBL_FONT_SIZE)

def one_col_n_row_grid(nz, z_bins, ylab, ymin, ymax, scale="log", xlab = r'$k$ [km/s]$^{-1}$', colormap=plt.cm.jet):
    # Set up plotting env
    fig = plt.figure(figsize=(5, nz))
    gs = gridspec.GridSpec(nz, 1, figure=fig, wspace=0.0, hspace=0.05)

    axs = [fig.add_subplot(gi) for gi in gs]

    axs[-1].set_xlabel(xlab, fontsize = AXIS_LBL_FONT_SIZE)

    plt.setp(axs[-1].get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)

    for ax in axs:
        plt.setp(ax.get_yticklabels(), fontsize = TICK_LBL_FONT_SIZE)
    axs[int(nz/2)].set_ylabel(ylab, fontsize = AXIS_LBL_FONT_SIZE)

    for i, ax in enumerate(axs):
        if i == nz:
            break
        ax.text(0.98, 0.94, "z=%.1f"%z_bins[i], transform=ax.transAxes, fontsize=TICK_LBL_FONT_SIZE, \
            verticalalignment='top', horizontalalignment='right', bbox={'facecolor':'white', 'pad':1})
        ax.set_yscale(scale)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xscale("log")
        ax.grid(True, which = 'major')
        ax.tick_params(which='major', direction='in', length=5, width=1)
        ax.tick_params(which='minor', direction='in', length=3, width=1)

    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible = False)

    # Set up colormap
    color_array=[colormap(i) for i in np.linspace(0, 1, nz)]

    return axs, color_array

def two_col_n_row_grid(nz, z_bins, ylab, ymin, ymax, scale="log", xlab = r'k [km/s]$^{-1}$', colormap=plt.cm.jet):
    # Set up plotting env
    fig = plt.figure(figsize=(10, nz/2))
    gs = gridspec.GridSpec(int((nz+1)/2), 2, figure=fig, wspace=0.01, hspace=0.05)
    # fig = plt.figure(figsize=(5, nz))
    # gs = gridspec.GridSpec(nz, 1, figure=fig, wspace=0.0, hspace=0.05)

    axs = [fig.add_subplot(gi) for gi in gs]

    axs[-1].set_xlabel(xlab, fontsize = AXIS_LBL_FONT_SIZE)
    axs[-2].set_xlabel(xlab, fontsize = AXIS_LBL_FONT_SIZE)

    plt.setp(axs[-1].get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
    plt.setp(axs[-2].get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)

    for ax in axs[0::2]:
        ax.set_ylabel(ylab, fontsize = AXIS_LBL_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize = TICK_LBL_FONT_SIZE)

    for i, ax in enumerate(axs):
        if i == nz:
            break
        ax.text(0.98, 0.94, "z=%.1f"%z_bins[i], transform=ax.transAxes, fontsize=TICK_LBL_FONT_SIZE, \
            verticalalignment='top', horizontalalignment='right', bbox={'facecolor':'white', 'pad':1})
        ax.set_yscale(scale)
        ax.set_xscale("log")
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.grid(True, which = 'major')
        ax.tick_params(which='major', direction='in', length=5, width=1)
        ax.tick_params(which='minor', direction='in', length=3, width=1)

    for ax in axs[:-2]:
        plt.setp(ax.get_xticklabels(), visible = False)

    for ax in axs[1::2]:
        plt.setp(ax.get_yticklabels(), visible = False)

    # Set up colormap
    color_array=[colormap(i) for i in np.linspace(0, 1, nz)]

    return axs, color_array

def create_tworow_figure(plt, nz, ratio_up2down, majorgrid=True, hspace=0, colormap=plt.cm.jet, ylim=0.05):
    fig = plt.figure()
    top_pos, bot_pos = gridspec.GridSpec(2, 1, height_ratios=[ratio_up2down, 1])
    top_ax = fig.add_subplot(top_pos)
    bot_ax = fig.add_subplot(bot_pos, sharex=top_ax)

    plt.setp(top_ax.get_xticklabels(), visible = False)

    bot_ax.grid(majorgrid, which = 'major')

    fig.subplots_adjust(hspace=hspace)

    color_array=[colormap(i) for i in np.linspace(0, 1, nz)]

    # Plot top axis
    set_topax_makeup(top_ax, majorgrid)
    bot_ax.set_xscale("log")

    bot_ax.set_ylim(-ylim, ylim)
    
    bot_ax.tick_params(which='major', direction='in', length=7, width=1)
    bot_ax.tick_params(which='minor', direction='in', length=4, width=0.8)

    bot_ax.set_xlabel(r'$k$ [km/s]$^{-1}$', fontsize = AXIS_LBL_FONT_SIZE)
    bot_ax.set_ylabel(r'$\Delta P/P_{\mathrm{t}}$', fontsize = AXIS_LBL_FONT_SIZE)

    plt.setp(bot_ax.get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
    plt.setp(bot_ax.get_yticklabels(), fontsize = TICK_LBL_FONT_SIZE)

    return top_ax, bot_ax, color_array

def plot_kPpi_zi(axi, k, p, e, ci):
    axi.errorbar(k, p*k/np.pi, xerr = 0, yerr = e*k/np.pi, fmt='o', color=ci, markersize=2)


def add_legend_no_error_bars(ax, location, ncol=1, bbox_to_anchor=(1,0.5)):
    # Get handles
    handles, labels = ax.get_legend_handles_labels()

    # Remove the errorbars
    handles = [h[0] for h in handles]

    ax.legend(handles, labels, loc = location, bbox_to_anchor = bbox_to_anchor, \
        fontsize='large', numpoints = 1, ncol=ncol, handletextpad=0.4)

class PowerPlotter(object):
    """docstring for PowerPlotter
    Attributes
    ----------
    karray
    zarray
    z_bins
    k_bins
    nk
    nz

    power_sp
    fid_power
    error
    """

    """
    By default true power is the fiducial, which can be zero.
    """
    def _readDBTFile(self, filename):
        try:
            power_table = ascii.read(filename, format='fixed_width')
            self.karray = np.array(power_table['kc'], dtype=np.double)
        except KeyError:
            power_table = ascii.read(filename)
            self.karray = np.array(power_table['kc'], dtype=np.double)

        # Set up z values and k values once by reading one file
        self.zarray = np.array(power_table['z'], dtype=np.double)

        self.z_bins = np.unique(self.zarray)
        self.k_bins = np.unique(self.karray)
        self.nz = self.z_bins.size
        self.nk = self.k_bins.size

        # Find out what kind of table we are reading
        # If it is QE result file
        if 'ThetaP' in power_table.colnames:
            thetap   = np.array(power_table['ThetaP'], dtype=np.double)
            self.fid_power = np.array(power_table['Pfid'], dtype=np.double)
            
            self.power_sp = np.split(self.fid_power + thetap, self.nz)
            self.error    = np.split(np.array(power_table['ErrorP'], dtype=np.double), self.nz)
            self.fid_power = np.split(self.fid_power, self.nz)
        # If it is FFT estimate file  
        elif 'P-FFT' in power_table.colnames:
            self.power_sp = np.split(np.array(power_table['P-FFT'], dtype=np.double), self.nz)
            self.error    = np.split(np.array(power_table['ErrorP-FFT'], dtype=np.double), self.nz)
            self.fid_power = np.zeros_like(self.power_sp)

        self.power_true = self.fid_power

    def __init__(self, filename):
        # Reading file into an ascii table  
        self._readDBTFile(filename)

    def addTruePowerFile(self, filename):
        try:
            power_true = np.load(filename[:-3]+"npy")
        except Exception as e:
            power_true_table = ascii.read(filename, format='fixed_width', guess=False)
            k_true           = np.unique(np.array(power_true_table['kc'], dtype=np.double))
            z_true           = np.unique(np.array(power_true_table['z'],  dtype=np.double))
            if 'P-FFT' in power_true_table.colnames:
                p_true = np.array(power_true_table['P-FFT'], dtype=np.double).reshape(len(z_true), len(k_true))
            elif 'P-ALN' in power_true_table.colnames:
                p_true = np.array(power_true_table['P-ALN'], dtype=np.double).reshape(len(z_true), len(k_true))
            elif 'Pfid' in power_true_table.colnames:
                p_true = np.array(power_true_table['Pfid'], dtype=np.double) + \
                np.array(power_true_table['ThetaP'], dtype=np.double)
                p_true = p_true.reshape(len(z_true), len(k_true))
            else:
                print("True power estimates cannot be read!")
                return -1

            interp_true      = RectBivariateSpline(z_true, k_true, p_true)

            self.power_true  = interp_true(self.z_bins, self.k_bins, grid=True)
            
            np.save(filename[:-4], power_true)

            del interp_true
            del power_true_table
            del p_true
            del k_true
            del z_true

    def plotRedshiftBin(self, nz, outplot_fname=None, two_row=False, pk_ymax=0.5, pk_ymin=1e-4, \
        rel_ylim=0.05, noise_dom=None, auto_ylim_xmin=-1, auto_ylim_xmax=1000, ignore_last_k_bins=-1):
        plt.clf()
        if two_row:
            top_ax, bot_ax = create_tworow_figure(plt, 1, 3, ylim=rel_ylim)[:-1]
        else:
            fig, top_ax = plt.subplots()
            set_topax_makeup(top_ax, ymin=pk_ymin, ymax=pk_ymax)
            plt.setp(top_ax.get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
            top_ax.set_xlabel(r'$k$ [km/s]$^{-1}$', fontsize = AXIS_LBL_FONT_SIZE)

        psz = self.power_sp[nz]
        erz = self.error[nz]
        ptz = self.power_true[nz]
        z_val = self.z_bins[nz]

        chi_sq_zb = (psz - ptz)**2 / erz**2
        
        if ignore_last_k_bins > 0:
            chi_sq_zb = chi_sq_zb[:-ignore_last_k_bins]
            ddof = self.k_bins.size-ignore_last_k_bins
        else:
            ddof =self.k_bins.size

        chi_sq_zb = np.sum(chi_sq_zb)

        # Start plotting
        top_ax.errorbar(self.k_bins, psz*self.k_bins/np.pi, xerr = 0, yerr = erz*self.k_bins/np.pi, \
            fmt='o', label="z=%.1f"%z_val, capsize=2, color='k')
        
        top_ax.errorbar(self.k_bins, ptz*self.k_bins/np.pi, xerr = 0, yerr = 0, \
            fmt=':', capsize=0, color='k')

        top_ax.text(0.05, 0.15, "z=%.1f"%z_val, transform=top_ax.transAxes, fontsize=TICK_LBL_FONT_SIZE, \
            verticalalignment='bottom', horizontalalignment='left', bbox={'facecolor':'white', 'pad':5})

        if noise_dom:
            top_ax.axvspan(noise_dom, self.k_bins[-1]*1.4, facecolor='0.5', alpha=0.5)
        
        if two_row:
            rel_err = psz / ptz  - 1
            bot_ax.errorbar(self.k_bins, rel_err, xerr = 0, yerr = erz/ptz, fmt='s:', \
                markersize=3, capsize=0, color='k')
            
            bot_ax.axvspan(noise_dom, self.k_bins[-1]*1.4, facecolor='0.5', alpha=0.5)

            rel_err = np.abs(rel_err) + erz/ptz
            rel_err = rel_err[np.logical_and(auto_ylim_xmin < self.k_bins, self.k_bins < auto_ylim_xmax)]
            yy = np.max(rel_err)
            bot_ax.set_ylim(-yy, yy)

        print("z={:.1f} Chi-Square / dof: {:.2f} / {:d}.".format(z_val, chi_sq_zb, ddof))

        if outplot_fname:
            plt.savefig(outplot_fname, dpi=300, bbox_inches='tight')

    def plotAll(self, outplot_fname=None, two_row=False, pk_ymax=0.5, pk_ymin=1e-4, rel_ylim=0.05, \
        colormap=plt.cm.jet, noise_dom=None, auto_ylim_xmin=-1, auto_ylim_xmax=1000, ignore_last_k_bins=-1):
        plt.clf()
        if two_row:
            top_ax, bot_ax, color_array = create_tworow_figure(plt, self.nz, 3, ylim=rel_ylim, colormap=colormap)
        else:
            fig, top_ax = plt.subplots()
            set_topax_makeup(top_ax, ymin=pk_ymin, ymax=pk_ymax)
            plt.setp(top_ax.get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
            top_ax.set_xlabel(r'$k$ [km/s]$^{-1}$', fontsize = AXIS_LBL_FONT_SIZE)
            color_array=[colormap(i) for i in np.linspace(0, 1, self.nz)]
        
        chi_sq = 0

        # Plot for each redshift bin
        for i in range(self.nz):
            psz = self.power_sp[i]
            erz = self.error[i]
            ptz = self.power_true[i]
            z_val = self.z_bins[i]
            ci = color_array[i]

            chi_sq_zb = (psz - ptz)**2 / erz**2

            if ignore_last_k_bins > 0:
                chi_sq_zb = chi_sq_zb[:-ignore_last_k_bins]

            chi_sq += np.sum(chi_sq_zb)

            top_ax.errorbar(self.k_bins, psz*self.k_bins/np.pi, xerr = 0, yerr = erz*self.k_bins/np.pi, \
                fmt='o', label="z=%.2f"%z_val, capsize=2, color=ci)
            
            top_ax.errorbar(self.k_bins, ptz*self.k_bins/np.pi, xerr = 0, yerr = 0, fmt=':', capsize=0, \
                color=ci)

            if two_row:
                bot_ax.errorbar(self.k_bins, psz / ptz  - 1, xerr = 0, yerr = erz/ptz, fmt='s:', \
                    markersize=3, capsize=0, color=ci)

        if two_row:
            rel_err = np.array(self.power_sp) / np.array(self.power_true)  - 1
            rel_err = np.abs(rel_err) + np.array(self.error)/np.array(self.power_true)
            rel_err = rel_err[:, np.logical_and(auto_ylim_xmin < self.k_bins, self.k_bins < auto_ylim_xmax)]
            yy = np.max(rel_err)
            bot_ax.set_ylim(-yy, yy)

        add_legend_no_error_bars(top_ax, "upper left", bbox_to_anchor=(1.03, 1))

        if noise_dom:
            top_ax.axvspan(noise_dom, self.k_bins[-1]*1.4, facecolor='0.5', alpha=0.5)
            if two_row:
                bot_ax.axvspan(noise_dom, self.k_bins[-1]*1.4, facecolor='0.5', alpha=0.5)

        if ignore_last_k_bins > 0:
            ddof = self.k_bins.size-ignore_last_k_bins
        else:
            ddof = self.k_bins.size
        
        print("Chi-Square / dof: {:.2f} / {:d}.".format(chi_sq, ddof*self.nz))

        if outplot_fname:
            plt.savefig(outplot_fname, dpi=300, bbox_inches='tight')
        
        





