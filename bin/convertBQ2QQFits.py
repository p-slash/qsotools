#!/usr/bin/env python
import argparse
import fitsio
import numpy as np
import healpy

from os.path import join as ospath_join
from os      import makedirs as os_makedirs

import qsotools.io as qio

if __name__ == '__main__':
    # Arguments passed to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="List of files. Starts with number of files.")
    parser.add_argument("OutputDir")
    parser.add_argument("--relative-dir", default=".", \
        help="Relative directory to fnames in filelist.")
    parser.add_argument("--nside", help="Heal pixel nside. Default: %(default)s", type=int, default=8)
    parser.add_argument("--read-coords", help="Reads coords from file. If not randomly generated.")
    parser.add_argument("--seed", help="Default: %(default)s", default=3434672, type=int)
    args = parser.parse_args()

    npixels = healpy.nside2npix(args.nside)
    RNST = np.random.default_rng(args.seed)

    with open(args.filelist, 'r') as file_qsolist:
        no_qsos = int(file_qsolist.readline())
        bq_fname_list = [ospath_join(args.relative_dir, x.rstrip()) \
            for x in file_qsolist]

    # The METADATA HDU contains a binary table with (at least) RA,DEC,Z,MOCKID
    metadata = np.zeros(no_qsos, dtype=[('RA', 'f8'), ('DEC', 'f8'), \
        ('Z', 'f8'), ('MOCKID', 'i8'), ('IPIX', 'i4')])
    metadata['MOCKID'] = np.arange(no_qsos)
    wavelength = []

    for imock in range(no_qsos):
        fl = bq_fname_list[imock]

        try:
            spectrum = qio.BinaryQSO(fl, 'r')
            if imock == 0:
                print(spectrum.wave.size)
                wavelength = np.array(spectrum.wave, dtype='float32')
        except Exception as e:
            print("Problem reading ", fl, flush=True)
            print(e)
            continue

        metadata['Z'][imock] = spectrum.z_qso
        if args.read_coords:
            metadata['RA'][imock] = spectrum.coord.ra.rad
            metadata['DEC'][imock] = spectrum.coord.dec.rad
        else:
            ra, dec = RNST.random(2) * [2, 1] * np.pi
            dec -= np.pi/2
            metadata['RA'][imock] = ra
            metadata['DEC'][imock] = dec

        ipix = healpy.ang2pix(args.nside, -metadata['DEC'][imock]+np.pi/2, metadata['RA'][imock])
        metadata['IPIX'][imock] = ipix

    # MOCKDIR/P/PIXNUM/lya-transmission-N-PIXNUM.fits
    os_makedirs(args.OutputDir, exist_ok=True)
    for ipix in range(npixels):
        P = int(ipix/100)
        meta1 = metadata[metadata['IPIX'] == ipix]
        ntemp = len(meta1['MOCKID'])
        
        if ntemp == 0:
            continue

        dir1 = ospath_join(args.OutputDir, f"{P}")
        dir2 = ospath_join(dir1, f"{ipix}")
        os_makedirs(dir1, exist_ok=True)
        os_makedirs(dir2, exist_ok=True)
        fname = ospath_join(dir2, f"lya-transmission-{args.nside}-{ipix}.fits")
        
        qqfile = qio.QQFile(fname)
        fluxes = np.zeros((ntemp, wavelength.size))
        
        for i1, imock in enumerate(meta1['MOCKID']):
            fl = bq_fname_list[imock]

            try:
                spectrum = qio.BinaryQSO(fl, 'r')
                fluxes[i1] = spectrum.flux
            except Exception as e:
                print("Problem reading ", fl, flush=True)
                print(e)
                continue

        qqfile.writeAll(meta1, wavelength, fluxes)











