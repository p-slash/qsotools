import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from   matplotlib        import gridspec
from   scipy.interpolate import RectBivariateSpline
from   astropy.io        import ascii

TICK_LBL_FONT_SIZE = 18
AXIS_LBL_FONT_SIZE = 20

def save_figure(outplot_fname, dpi=200):
    if outplot_fname:
        plt.savefig(outplot_fname, dpi=dpi, bbox_inches='tight')

def ticks_makeup(ax):
    ax.tick_params(direction='in', which='major', length=7, width=1, right=True,  top=True)
    ax.tick_params(direction='in', which='minor', length=4, width=1, right=True,  top=True)
    
    plt.setp(ax.get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
    plt.setp(ax.get_yticklabels(), fontsize = TICK_LBL_FONT_SIZE)

def set_topax_makeup(top_ax, majorgrid=True, ymin=None, ymax=None):
    top_ax.grid(majorgrid, which='major')
    top_ax.set_yscale("log")
    top_ax.set_xscale("log")
    
    if ymin:
        top_ax.set_ylim(ymin=ymin)
    if ymax:
        top_ax.set_ylim(ymax=ymax)

    ticks_makeup(top_ax)

    top_ax.set_ylabel(r'$kP/\pi$', fontsize = AXIS_LBL_FONT_SIZE)

def one_col_n_row_grid(nz, z_bins, ylab, ymin, ymax, scale="log", \
    xlab = r'$k$ [s$\,$km$^{-1}$]', colormap=plt.cm.jet):
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
        ax.text(0.98, 0.94, "z=%.1f"%z_bins[i], transform=ax.transAxes, \
            fontsize=TICK_LBL_FONT_SIZE, verticalalignment='top', horizontalalignment='right', \
            bbox={'facecolor':'white', 'pad':1})
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

def two_col_n_row_grid(nz, z_bins, ylab, ymin, ymax, scale="log", \
    xlab = r'$k$ [s$\,$km$^{-1}$]', colormap=plt.cm.jet):
    # Set up plotting env
    fig = plt.figure(figsize=(10, nz/2))
    gs = gridspec.GridSpec(int((nz+1)/2), 2, figure=fig, wspace=0.01, hspace=0.05)

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

def create_tworow_figure(nz, ratio_up2down, majorgrid=True, hspace=0, \
    colormap=plt.cm.jet, ylim=0.05):
    fig = plt.figure()
    top_pos, bot_pos = gridspec.GridSpec(2, 1, height_ratios=[ratio_up2down, 1])
    top_ax = fig.add_subplot(top_pos)
    bot_ax = fig.add_subplot(bot_pos, sharex=top_ax)

    plt.setp(top_ax.get_xticklabels(), visible = False)

    bot_ax.grid(majorgrid, which = 'major')

    fig.subplots_adjust(hspace=hspace)

    color_array=[colormap(i) for i in np.linspace(0, 1, nz)]

    # Plot top axis
    # set_topax_makeup(top_ax, majorgrid)
    bot_ax.set_xscale("log")

    bot_ax.set_ylim(-ylim, ylim)
    
    ticks_makeup(bot_ax)

    bot_ax.set_xlabel(r'$k$ [s$\,$km$^{-1}$]', fontsize = AXIS_LBL_FONT_SIZE)
    bot_ax.set_ylabel(r'$\Delta P/P_{\mathrm{t}}$', fontsize = AXIS_LBL_FONT_SIZE)

    return top_ax, bot_ax, color_array

def add_legend_no_error_bars(ax, location="center left", ncol=1, bbox_to_anchor=(1.03,0.5), \
    fontsize='large'):
    # Get handles
    handles, labels = ax.get_legend_handles_labels()

    # Remove the errorbars
    handles = [h[0] for h in handles]

    ax.legend(handles, labels, loc = location, bbox_to_anchor = bbox_to_anchor, \
        fontsize = fontsize, numpoints = 1, ncol = ncol, handletextpad = 0.4)

class PowerPlotter(object):
    """PowerPlotter is an object to plot QMLE power spectrum results by individual redshift bins or all in one.
    Simply initialize with a filename.

    Attributes
    ----------
    karray
    zarray
    z_bins
    k_bins
    nk
    nz

    power_qmle
    power_fid
    power_true
    error
    fisher : initialized to diagonal of inverse errors
    """

    def _autoRelativeYLim(self, ax, rel_err, erz, ptz, auto_ylim_xmin, auto_ylim_xmax):
        rel_err = np.abs(rel_err) + erz/ptz
        autolimits = np.logical_and(auto_ylim_xmin < self.k_bins, self.k_bins < auto_ylim_xmax)
        if rel_err.ndim == 1:
            rel_err = rel_err[autolimits]
        else:
            rel_err = rel_err[:, autolimits]
        yy = np.max(rel_err)
        ax.set_ylim(-yy, yy)

    def _readDBTFile(self, filename):
        """Set up attributes. By default true power is the fiducial, which can be zero."""
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

        # Read k edges
        k1 = np.unique(np.array(power_table['k1'], dtype=np.double))
        k2 = np.unique(np.array(power_table['k2'], dtype=np.double))
        self.k_edges = np.append(k1, k2[-1])

        # Find out what kind of table we are reading
        # If it is QE result file
        if 'ThetaP' in power_table.colnames:
            thetap   = np.array(power_table['ThetaP'], dtype=np.double)
            self.power_fid = np.array(power_table['Pfid'], dtype=np.double)
            
            self.power_qmle = np.reshape(self.power_fid + thetap, (self.nz, self.nk))
            self.error    = np.array(power_table['ErrorP'], dtype=np.double).reshape((self.nz, self.nk))
            
            self.power_qmle_full  = np.array(power_table['d'], dtype=np.double).reshape((self.nz, self.nk))
            self.power_qmle_noise = np.array(power_table['b'], dtype=np.double).reshape((self.nz, self.nk))
            self.power_qmle_fid   = np.array(power_table['t'], dtype=np.double).reshape((self.nz, self.nk))

            self.power_fid = self.power_fid.reshape((self.nz, self.nk))
        # If it is FFT estimate file  
        elif 'P-FFT' in power_table.colnames:
            self.power_qmle = np.array(power_table['P-FFT'], dtype=np.double).reshape((self.nz, self.nk))
            self.error    = np.array(power_table['ErrorP-FFT'], dtype=np.double).reshape((self.nz, self.nk))
            self.power_fid = np.zeros_like(self.power_qmle)

        self.power_true = self.power_fid

    def __init__(self, filename):
        # Reading file into an ascii table  
        self._readDBTFile(filename)
        self.fisher = np.diag(1/self.error.ravel()**2)
        print("There are {:d} redshift bins and {:d} k bins.".format(self.nz, self.nk))

    def setFisher(self, fisher):
        self.fisher = fisher

    def addTruePowerFile(self, filename):
        """Sets true power from given file. Saves it as .npy for future readings."""
        try:
            power_true = np.load(filename[:-3]+"npy")
        except Exception as e:
            power_true_table = ascii.read(filename, format='fixed_width', guess=False)
            k_true           = np.unique(np.array(power_true_table['kc'], dtype=np.double))
            z_true           = np.unique(np.array(power_true_table['z'],  dtype=np.double))
            if 'P-FFT' in power_true_table.colnames:
                p_true = np.array(power_true_table['P-FFT'], dtype=np.double).\
                    reshape(len(z_true), len(k_true))
            elif 'P-ALN' in power_true_table.colnames:
                p_true = np.array(power_true_table['P-ALN'], dtype=np.double).\
                    reshape(len(z_true), len(k_true))
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

    def plotRedshiftBin(self, nz, outplot_fname=None, two_row=False, plot_true=True, \
        plot_dbt=False, rel_ylim=0.05, noise_dom=None, auto_ylim_xmin=-1, \
        auto_ylim_xmax=1000, kmax_chisquare=None):
        """Plot QMLE results for given redshift bin nz.

        Parameters
        ----------
        nz : int
            Redshift bin number ranges from 0 to self.nz-1.
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        two_row : bool, optional
            When passed, add a lower panel for relative error computed by using the true power.
        plot_true : bool, optional
            Plot true value if True.
        plot_dbt : bool, optiona
            Plot full power and noise estimate individually.
        rel_ylim : float, optional
            Y axis limits for the relative error on the lower panel.
        noise_dom : float, optional
            Adds a shade for k larger than this value.
        auto_ylim_xmin, auto_ylim_xmax : float, optional
            Automatically scales the relative error panel by limiting the axis range between these values.
        kmax_chisquare : float, optional
            When passed ignore k>kmax_chisquare modes from the chi square.
        """
        if two_row:
            top_ax, bot_ax = create_tworow_figure(1, 3, ylim=rel_ylim)[:-1]
            set_topax_makeup(top_ax)
            plt.setp(top_ax.get_xticklabels(), visible=False)
        else:
            fig, top_ax = plt.subplots()
            set_topax_makeup(top_ax)
            plt.setp(top_ax.get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
            top_ax.set_xlabel(r'$k$ [s$\,$km$^{-1}$]', fontsize = AXIS_LBL_FONT_SIZE)

        if plot_dbt:
            psz = self.power_qmle_full[nz]
            pnz = self.power_qmle_noise[nz]
            psz_label = "Raw"
        else:
            psz = self.power_qmle[nz]
            psz_label = "Est"

        erz = self.error[nz]
        ptz = self.power_true[nz]
        z_val = self.z_bins[nz]

        # Start plotting
        top_ax.errorbar(self.k_bins, psz*self.k_bins/np.pi, xerr = 0, yerr = erz*self.k_bins/np.pi, \
            fmt='o', label=psz_label, markersize=3, capsize=2, color='k')
        
        if plot_dbt:
            top_ax.errorbar(self.k_bins, pnz*self.k_bins/np.pi, xerr = 0, yerr = 0, \
                fmt='s', label="Noise", markersize=3, capsize=0, color='r')
            top_ax.legend(fontsize = 'large')
        
        if plot_true:
            top_ax.errorbar(self.k_bins, ptz*self.k_bins/np.pi, xerr = 0, yerr = 0, \
                fmt=':', capsize=0, color='k')

        top_ax.text(0.9, 0.9, "z=%.1f"%z_val, transform=top_ax.transAxes, fontsize=TICK_LBL_FONT_SIZE, \
            verticalalignment='top', horizontalalignment='right', bbox={'facecolor':'white', 'pad':4})

        if noise_dom:
            top_ax.set_xlim(xmax=self.k_bins[-1]*1.1)
            top_ax.axvspan(noise_dom, self.k_bins[-1]*1.1, facecolor='0.5', alpha=0.5)

            if two_row:
                bot_ax.axvspan(noise_dom, self.k_bins[-1]*1.1, facecolor='0.5', alpha=0.5)
        
        if two_row:
            rel_err = psz / ptz  - 1
            bot_ax.errorbar(self.k_bins, rel_err, xerr = 0, yerr = erz/ptz, fmt='s:', \
                markersize=3, capsize=0, color='k')

            self._autoRelativeYLim(bot_ax, rel_err, erz, ptz, auto_ylim_xmin, auto_ylim_xmax)

        pkpi = psz*self.k_bins/np.pi
        ymin = np.min(pkpi[self.k_bins<auto_ylim_xmax])
        top_ax.set_ylim(ymin=ymin)
        
        yticks = []
        if ymin < 1e-3:
            yticks.append(1e-3)
        yticks.append(1e-2)
        if np.max(pkpi) > 0.1:
            yticks.append(1e-1)

        top_ax.set_yticks(yticks)

        chi2, ddof = self.getChiSquare(nz, kmax=kmax_chisquare)
        print("z={:.1f} Chi-Square / dof: {:.2f} / {:d}.".format(z_val, chi2, ddof))

        save_figure(outplot_fname)

    def plotAll(self, outplot_fname=None, two_row=False, plot_true=True, pk_ymax=0.5, \
        pk_ymin=1e-4, rel_ylim=0.05, colormap=plt.cm.jet, noise_dom=None, \
        fmt=None, auto_ylim_xmin=-1, auto_ylim_xmax=1000, kmax_chisquare=None):
        """Plot QMLE results for all redshifts in one figure.

        Parameters
        ----------
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        two_row : bool, optional
            When passed, add a lower panel for relative error computed by using the true power.
        pk_ymax, pk_ymin : float, optional
            Maximum and minimum y axis limits for kP/pi.
        fmt: str, optional
            Define fmt for top axis errorbar plot. Default is "o"
        rel_ylim : float, optional
            Y axis limits for the relative error on the lower panel.
        colormap : plt.cm, optional
            Colormap to use for different redshift bins.
        noise_dom : float, optional
            Adds a shade for k larger than this value.
        auto_ylim_xmin, auto_ylim_xmax : float, optional
            Automatically scales the relative error panel by limiting the axis range between these values.
        kmax_chisquare : float, optional
            When passed ignore k>kmax_chisquare modes from the chi square.
        """
        if two_row:
            top_ax, bot_ax, color_array = create_tworow_figure(self.nz, 3, ylim=rel_ylim, \
                colormap=colormap)
            plt.setp(top_ax.get_xticklabels(), visible=False)
        else:
            fig, top_ax = plt.subplots()
            plt.setp(top_ax.get_xticklabels(), fontsize = TICK_LBL_FONT_SIZE)
            top_ax.set_xlabel(r'$k$ [s$\,$km$^{-1}$]', fontsize = AXIS_LBL_FONT_SIZE)
            color_array=[colormap(i) for i in np.linspace(0, 1, self.nz)]

        set_topax_makeup(top_ax, ymin=pk_ymin, ymax=pk_ymax)

        if fmt is None:
            fmt = "o"

        # Plot for each redshift bin
        for i in range(self.nz):
            psz = self.power_qmle[i]
            erz = self.error[i]
            ptz = self.power_true[i]
            z_val = self.z_bins[i]
            ci = color_array[i]

            top_ax.errorbar(self.k_bins, psz*self.k_bins/np.pi, yerr=erz*self.k_bins/np.pi, \
                fmt=fmt, label="z=%.1f"%z_val, markersize=3, capsize=2, color=ci)
            
            if plot_true:
                top_ax.errorbar(self.k_bins, ptz*self.k_bins/np.pi, fmt=':', capsize=0, color=ci)

            if two_row:
                bot_ax.errorbar(self.k_bins, psz / ptz  - 1, xerr = 0, yerr = erz/ptz, fmt='s--', \
                    markersize=3, capsize=0, color=ci)

        if two_row:
            rel_err = self.power_qmle / self.power_true  - 1
            self._autoRelativeYLim(bot_ax, rel_err, self.error, self.power_true, \
                auto_ylim_xmin, auto_ylim_xmax)

        add_legend_no_error_bars(top_ax, "upper left", bbox_to_anchor=(1.0, 1.03))

        if noise_dom:
            top_ax.set_xlim(xmax=self.k_bins[-1]*1.1)
            top_ax.axvspan(noise_dom, self.k_bins[-1]*1.1, facecolor='0.5', alpha=0.5)
            if two_row:
                bot_ax.axvspan(noise_dom, self.k_bins[-1]*1.1, facecolor='0.5', alpha=0.5)
        
        chi2, ddof = self.getChiSquare(kmax=kmax_chisquare)
        print("Chi-Square / dof: {:.2f} / {:d}.".format(chi2, ddof))

        save_figure(outplot_fname)

    def plotMultiDeviation(self, outplot_fname=None, two_col=False, rel_ylim=0.05, \
        colormap=plt.cm.jet, noise_dom=None, auto_ylim_xmin=-1, auto_ylim_xmax=1000):
        """Plot QMLE relative errors with each redshift bin having its own panel.

        Parameters
        ----------
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        two_col : bool, optional
            When passed, creates a two-column plot.
        rel_ylim : float, optional
            Y axis limits for the relative error on the lower panel.
        colormap : plt.cm, optional
            Colormap to use for different redshift bins.
        noise_dom : float, optional
            Adds a shade for k larger than this value.
        auto_ylim_xmin, auto_ylim_xmax : float, optional
            Automatically scales the relative error panel by limiting the axis range between these values.
        """
        # Plot one column
        if two_col:
            axs, color_array = two_col_n_row_grid(self.nz, self.z_bins, ylab=r'$\Delta P/P_{\mathrm{t}}$', \
                ymin=-rel_ylim, ymax=rel_ylim, scale='linear', colormap=colormap)
        else:
            axs, color_array = one_col_n_row_grid(self.nz, self.z_bins, ylab=r'$\Delta P/P_{\mathrm{t}}$', \
                ymin=-rel_ylim, ymax=rel_ylim, scale='linear', colormap=colormap)

        # Plot for each redshift bin
        for i in range(self.nz):
            psz = self.power_qmle[i]
            erz = self.error[i]
            ptz = self.power_true[i]
            z_val = self.z_bins[i]
            ci = color_array[i]

            rel_err = psz / ptz  - 1

            axs[i].errorbar(self.k_bins, rel_err, xerr = 0, yerr = erz/ptz, \
                fmt='o', color=ci, markersize=2)

            axs[i].axhline(color='k')
            
            if noise_dom:
                axs[i].set_xlim(xmax=self.k_bins[-1]*1.1)
                axs[i].axvspan(noise_dom, self.k_bins[-1]*1.1, facecolor='0.5', alpha=0.5)

            self._autoRelativeYLim(axs[i], rel_err, erz, ptz, auto_ylim_xmin, auto_ylim_xmax)

        save_figure(outplot_fname)

    def getChiSquare(self, zbin=None, fisher=None, kmin=None, kmax=None):
        if fisher:
            invcov = fisher
        else:
            invcov = self.fisher

        d = (self.power_qmle-self.power_true).ravel()
        to_remove = np.zeros_like(d, dtype=np.bool)

        if kmax and kmax > 0:
            to_remove |= (self.karray > kmax)
        if kmin and kmin > 0:
            to_remove |= (self.karray < kmin)
        
        if zbin:
            to_remove |= (self.zarray != self.z_bins[zbin])

        if to_remove.any():
            invcov = np.delete(invcov, to_remove, axis=0) 
            invcov = np.delete(invcov, to_remove, axis=1)
            d = np.delete(d, to_remove, axis=0)

        return d@invcov@d, d.size

class FisherPlotter(object):
    """FisherPlotter is object to plot the Fisher matrix in its entirety or in individual k & z bins.
    You need to initialize with number of redshift bins (nz), first redshift bin (z1) and bin width (dz).
    
    Parameters
    ----------
    filename : str
        Filename for the Fisher matrix..
    nz : int, optional
        Number of redshift bins. Default is None.
    dz : float, optional
        Redshift bin width. Default is 0.2.
    z1 : float, optional
        First redshift bin. Default is 1.8.
    skiprows : int, optional
        Number of rows to skip when reading fisher matrix file. Default is 1.

    __init__(filename, nz=None, dz=0.2, z1=1.8, skiprows=1)
        Reads the Fisher matrix into fisher and computes its normalization. If nz is passed, sets up 
        nz, nk and zlabels.

    Attributes
    ----------
    fisher
    invfisher
    nz
    nk
    zlabels

    """
    def __init__(self, filename, k_edges, nz, dz=0.2, z1=1.8, skiprows=1):
        self.fisher = np.loadtxt(filename, skiprows=skiprows)
        self.k_edges = k_edges
        self.nz = nz
        self.dz = dz
        self.z1 = z1
        self.nk = int(self.fisher.shape[0]/nz)
        self.zlabels = ["%.1f" % z for z in z1 + np.arange(nz) * dz]

        try:
            self.invfisher = np.linalg.inv(self.fisher)
        except Exception as e:
            self.invfisher = None
            print("Cannot invert the Fisher matrix.")

    def _setTicks(self, ax):
        if self.nz:
            ax.xaxis.set_major_locator(ticker.FixedLocator(self.nk*np.arange(self.nz)))
            ax.xaxis.set_minor_locator(ticker.FixedLocator(self.nk*(np.arange(self.nz) +0.5)))

            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.xaxis.set_minor_formatter(ticker.FixedFormatter(self.zlabels))

            for tick in ax.xaxis.get_minor_ticks():
                tick.tick1line.set_markersize(0)
                tick.tick2line.set_markersize(0)
                tick.label1.set_horizontalalignment('center')

            ax.yaxis.set_major_locator(ticker.FixedLocator(self.nk*np.arange(self.nz)))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(self.nk*(np.arange(self.nz) +0.5)))
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(ticker.FixedFormatter(self.zlabels))

            for tick in ax.yaxis.get_minor_ticks():
                tick.tick1line.set_markersize(0)
                tick.tick2line.set_markersize(0)
                tick.label1.set_horizontalalignment('right')

            ax.xaxis.set_tick_params(labelsize=TICK_LBL_FONT_SIZE)
            ax.yaxis.set_tick_params(labelsize=TICK_LBL_FONT_SIZE)

    def _setScale(self, matrix, scale, Ftxt='F', Fsub='\\alpha'):
        if scale == "norm":
            cbarlbl = r"$%s_{%s%s'}/\sqrt{%s_{%s%s}%s_{%s'%s'}}$" \
                % (Ftxt, Fsub, Fsub, Ftxt,Fsub, Fsub, Ftxt, Fsub, Fsub)
            
            fk_v = np.sqrt(matrix.diagonal())
            norm = np.outer(fk_v, fk_v)
            grid = matrix/norm
            colormap = plt.cm.seismic
        elif scale == "log":
            cbarlbl = r"$\log %s_{%s%s'}$" % (Ftxt, Fsub, Fsub)

            grid = np.log10(matrix)
            colormap = plt.cm.BuGn
        else:
            cbarlbl = r"$%s_{%s%s'}$" % (Ftxt, Fsub, Fsub)
            grid = matrix
            colormap = plt.cm.BuGn

        return grid, cbarlbl, colormap

    def plotAll(self, scale="norm", outplot_fname=None, inv=False, **kwargs):
        """Plot the entire Fisher matrix or its inverse.

        Parameters
        ----------
        scale : str, optional
            To normalize with respect to the diagonal pass "norm". 
            To plot log10 pass "log". Anything else leaves the Fisher matrix as it is. 
            Default is "norm".
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        inv : bool, optional
            Plot the inverse instead.
        kwargs: ** for imshow
        """
        fig, ax = plt.subplots()
        Ftxt = "F" if not inv else "F^{-1}"
        
        if self.invfisher is None and inv:
            print("Fisher is not invertable.")
            exit(1)
        if inv:
            tmp = self.invfisher
        else:
            tmp = self.fisher

        grid, cbarlbl, colormap = self._setScale(tmp, scale, Ftxt)

        im = ax.imshow(grid, cmap=colormap, origin='upper', \
            extent=[0, self.fisher.shape[0], self.fisher.shape[0], 0], \
            **kwargs)

        self._setTicks(ax)

        ax.grid(color='k', alpha=0.3)

        cbar = fig.colorbar(im)
        cbar.set_label(cbarlbl, fontsize=AXIS_LBL_FONT_SIZE)

        save_figure(outplot_fname)

    def _plotOneBin(self, x_corners, data, cbarlbl, axis_lbl, cmap, ticks, **kwargs):
        fig, ax = plt.subplots()

        im = ax.pcolormesh(x_corners, x_corners, data, cmap=cmap, shading='flat', **kwargs)
        if x_corners[-1]/x_corners[0] > 10:
            ax.set_xscale("log")
            ax.set_yscale("log")
        
        ax.set_xlim(xmax=x_corners[-1])
        ax.set_ylim(ymax=x_corners[-1])

        cbar = fig.colorbar(im, ticks=np.linspace(-1, 1, 6))
        cbar.set_label(cbarlbl, fontsize = AXIS_LBL_FONT_SIZE)
        cbar.ax.tick_params(labelsize = TICK_LBL_FONT_SIZE)

        ax.grid(color='k', alpha=0.3)
        plt.xticks(fontsize = TICK_LBL_FONT_SIZE)
        plt.yticks(fontsize = TICK_LBL_FONT_SIZE)
        
        ax.set_xlabel(axis_lbl, fontsize=AXIS_LBL_FONT_SIZE)
        ax.set_ylabel(axis_lbl, fontsize=AXIS_LBL_FONT_SIZE)
        
        ticks_makeup(ax)

        if ticks is not None:
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

        return ax

    def plotKBin(self, kb, scale="norm", ticks=None, inv=False, outplot_fname=None, \
        colormap=plt.cm.RdBu_r, **kwargs):
        """Plot Fisher matrix for a given k bin, i.e. redshift correlations.

        Parameters
        ----------
        kb : int
            Bin number for k to plot.
        scale : str, optional
            To normalize with respect to the diagonal pass "norm". 
            To plot log10 pass "log". Anything else leaves the Fisher matrix as it is. 
            Default is "norm".
        ticks : list or np.array
            Default is None, so automated.
        inv : bool, optional
            Plot the inverse instead.
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        colormap : plt.cm, optional
            Colormap to use for scale. Default is RdBu_r.
        kwargs: ** for imshow
        """
        Ftxt = "F" if not inv else "F^{-1}"
        
        if self.invfisher is None and inv:
            print("Fisher is not invertable.")
            exit(1)
        if inv:
            tmp = self.invfisher
        else:
            tmp = self.fisher

        grid, cbarlbl, _ = self._setScale(tmp, scale, Ftxt, Fsub='z')

        zbyz_corr = grid[kb::self.nk, :]
        zbyz_corr = zbyz_corr[:, kb::self.nk]
        
        zcoords = np.arange(self.nz+1) * self.dz + self.z1 - self.dz/2

        ax = self._plotOneBin(zcoords, zbyz_corr, cbarlbl, r"$z$", colormap, \
            ticks, **kwargs)

        save_figure(outplot_fname)

    def plotZBin(self, zb, scale="norm", ticks=None, inv=False, outplot_fname=None, \
        colormap=plt.cm.RdBu_r, **kwargs):
        """Plot Fisher matrix for a given z bin, i.e. k correlations.

        Parameters
        ----------
        zb : int
            Bin number for redshift to plot.
        scale : str, optional
            To normalize with respect to the diagonal pass "norm". 
            To plot log10 pass "log". Anything else leaves the Fisher matrix as it is. 
            Default is "norm".
        ticks : list or np.array
            Default is None, so automated.
        inv : bool, optional
            Plot the inverse instead.
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        colormap : plt.cm, optional
            Colormap to use for scale. Default is RdBu_r.
        """
        Ftxt = "F" if not inv else "F^{-1}"
        
        if self.invfisher is None and inv:
            print("Fisher is not invertable.")
            exit(1)
        if inv:
            tmp = self.invfisher
        else:
            tmp = self.fisher

        grid, cbarlbl, _ = self._setScale(tmp, scale, Ftxt, Fsub='k')

        kbyk_corr = grid[self.nk*zb:self.nk*(zb+1), :]
        kbyk_corr = kbyk_corr[:, self.nk*zb:self.nk*(zb+1)]

        ax = self._plotOneBin(self.k_edges, kbyk_corr, cbarlbl, r"$k$ [s$\,$km$^{-1}$]", \
            colormap, ticks, **kwargs)

        save_figure(outplot_fname)

    def plotKCrossZbin(self, zb, scale="norm", ticks=None, inv=False, outplot_fname=None, \
        colormap=plt.cm.RdBu_r, **kwargs):
        """Plot Fisher matrix for a given (z, z+1) pair, i.e. k correlations cross z bins.

        Parameters
        ----------
        zb : int
            Bin number for redshift to plot. The figure is zb, zb+1 correlations.
        scale : str, optional
            To normalize with respect to the diagonal pass "norm". 
            To plot log10 pass "log". Anything else leaves the Fisher matrix as it is. 
            Default is "norm".
        ticks : list or np.array
            Default is None, so automated.
        inv : bool, optional
            Plot the inverse instead.
        outplot_fname : str, optional
            When passed, figure is saved with this filename.
        colormap : plt.cm, optional
            Colormap to use for scale. Default is RdBu_r.
        kwargs: ** for imshow
        """
        Ftxt = "F" if not inv else "F^{-1}"
        
        if self.invfisher is None and inv:
            print("Fisher is not invertable.")
            exit(1)
        if inv:
            tmp = self.invfisher
        else:
            tmp = self.fisher

        grid, cbarlbl, _ = self._setScale(tmp, scale, Ftxt, Fsub='k')

        next_z_bin = zb+1
        if next_z_bin >= self.nz:
            return
        
        kbyk_corr = grid[self.nk*zb:self.nk*(zb+1), :]
        kbyk_corr = kbyk_corr[:, self.nk*next_z_bin:self.nk*(next_z_bin+1)]

        ax = self._plotOneBin(self.k_edges, kbyk_corr, cbarlbl, r"$k$ [s$\,$km$^{-1}$]", \
            colormap, ticks, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")

        save_figure(outplot_fname)









