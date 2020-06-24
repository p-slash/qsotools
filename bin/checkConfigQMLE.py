#!/usr/bin/env python
import argparse
from os.path import join as ospath_join

import numpy as np
import matplotlib.pyplot as plt

import qsotools.io as qio
import qsotools.fiducial as fid
import qsotools.specops as so

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", help="Config file")
    parser.add_argument("--plotfname", "-pf", help="Filename to save the plot.")
    parser.add_argument("--increase-z-res", "-mz", type=int, default=2, \
        help="Divide z bin width by this integer")
    args = parser.parse_args()

    config_qmle = qio.ConfigQMLE(args.ConfigFile)
    
    hist_nz = config_qmle.z_n * args.increase_z_res
    hist_dz = config_qmle.z_d / args.increase_z_res
    z_edges = config_qmle.z_0 + hist_dz * (np.arange(hist_nz+1)-0.5)
    # Add overflow bins
    z_edges = np.append(z_edges, 10)
    z_edges = np.insert(z_edges, 0, 0)

    pixel_z_hist = np.zeros(hist_nz+2)

    min_dv = []
    max_dv = []

    file_list = open(config_qmle.qso_list, 'r')
    header = file_list.readline()
    for fl in file_list:
        f = ospath_join(config_qmle.qso_dir, fl.rstrip())
        bq = qio.BinaryQSO(f, 'r')
        bq.read()
        
        # Compute min dv, max dv and append to list
        v_arr = fid.LIGHT_SPEED * np.log(bq.wave)
        vmin = np.min(np.diff(v_arr))
        vmax = v_arr[-1] - v_arr[0]
        min_dv.append(vmin)
        max_dv.append(vmax)

        # Add to pixel z histogram
        pixel_z_hist += so.getStats(bq.wave, bq.flux, bq.error, z_edges)[1]

    if pixel_z_hist[0] > 0:
        print("Warning: Some pixels are below the redshift range."\
            " This can cause problems in the Fisher Matrix.")
    if pixel_z_hist[-1] > 0:
        print("Warning: Some pixels are above the redshift range."\
            " This can cause problems in the Fisher Matrix.")

    # Plot histogram of pixel redshifts
    if args.plotfname:
        barws = np.diff(z_edges)
        barws[1:-1]*=0.8
        z_centers = (z_edges[1:]+z_edges[:-1])/2
        plt.bar(z_centers, pixel_z_hist, width=barws, \
            align='center', linewidth=1, ec='k', alpha=0.7)
        plt.xlim(z_edges[1]-hist_dz, z_edges[-2]+hist_dz)
        plt.grid(True, which='major', alpha=0.3)
        plt.xlabel(r'$z$')
        plt.ylabel('Count')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.title('Histogram of Pixel Redshifts')
        plt.savefig(args.plotfname, dpi=300, bbox_inches="tight")

    minmin_dv = np.min(min_dv)
    meanmin_dv = np.mean(min_dv)
    k_nyq = np.pi/minmin_dv
    print(("Minimum pixel spacing is {:.2f} km/s"\
        ", which corresponds to k_Nyquist={:.2e} s/km").format(minmin_dv, k_nyq))
    print(("Mean between spectra {:.2f} km/s"\
        ", which corresponds to <k_Nyq>={:.2e} s/km").format(meanmin_dv, np.pi/meanmin_dv))
    
    maxmax_dv = np.max(max_dv)
    meanmax_dv = np.mean(max_dv)
    print(("Maximum pixel separation is {:.2f} km/s"\
        ", which corresponds to k_Fund={:.2e} s/km").format(maxmax_dv, 2*np.pi/maxmax_dv))
    print(("Mean between spectra is {:.2f} km/s"\
        ", which corresponds to k_Fund={:.2e} s/km").format(meanmax_dv, 2*np.pi/meanmax_dv))
    
    # Check if the last k bin is empty
    if config_qmle.k_edges[-1] > k_nyq:
        print(f"{bcolors.BOLD}Warning: The last k edge is above the Nyquist.{bcolors.ENDC}")
        if config_qmle.k_edges[-2] > k_nyq:
            print(f"{bcolors.FAIL}{bcolors.BOLD}Error: The last k bin is empty!{bcolors.ENDC}")
        else:
            print("But last bin is not empy.")
    else:
        print(f"{bcolors.OKBLUE}{bcolors.BOLD}Passed: k bins end before Nyquist.{bcolors.ENDC}")

    # Check if config_qmle.sq_vlength > max dv
    if config_qmle.sq_vlength < maxmax_dv:
        print(f"{bcolors.FAIL}{bcolors.BOLD}Error: SQ table does not cover {bcolors.ENDC}"\
            f"{bcolors.FAIL}{bcolors.BOLD}the entire velocity separation.{bcolors.ENDC}")
        print(f"{bcolors.BOLD}Increase VelocityLength to at least {maxmax_dv:.1f}.{bcolors.ENDC}")











