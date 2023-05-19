import argparse
import glob
from multiprocessing import Pool
from os.path import (
    join as ospath_join,
    dirname as ospath_dir
)
import logging

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve
import fitsio

from tqdm import tqdm


@njit("f8[:, :](i8, f8[:])")
def _fast_construct_fisher(ndim, upper_fisher):
    fisher = np.empty((ndim, ndim))
    fari = 0
    for did in np.arange(ndim):
        for jj in range(ndim - did):
            fisher[jj, jj + did] = upper_fisher[fari + jj]
            fisher[jj + did, jj] = upper_fisher[fari + jj]

        fari += ndim - did

    return fisher


class Chunk():
    def __init__(self, data, ndim, istart):
        self.ndim = ndim
        self.istart = istart
        self.pk = data[:ndim] - data[ndim:2 * ndim] - data[2 * ndim:3 * ndim]
        self.upper_fisher = data[3 * ndim:]

    def form_square_fisher(self):
        return _fast_construct_fisher(self.ndim, self.upper_fisher)

    def add_to_total_fisher(self, total_fisher, m=1):
        fisher = self.form_square_fisher()
        s1 = np.s_[self.istart:self.istart + self.ndim]
        total_fisher[s1, s1] += m * fisher
        return total_fisher

    def add_to_total_power(self, total_power, m=1):
        s1 = np.s_[self.istart:self.istart + self.ndim]
        total_power[s1] += m * self.pk
        return total_power


def readOneFitsFile(fname):
    f = fitsio.FITS(fname)
    chunks = []

    for hdu in f[1:]:
        hdr = hdu.read_header()
        data = hdu.read()

        ndim = hdr['NQDIM']
        istart = hdr['ISTART']
        chunks.append(Chunk(data, ndim, istart))

    f.close()
    return chunks


def calc_total_ps_fisher(chunks, nk, nz):
    ntot = nk * nz
    fisher = np.zeros((ntot, ntot))
    power = np.zeros(ntot)

    for chunk in chunks:
        fisher = chunk.add_to_total_fisher(fisher)
        power = chunk.add_to_total_power(power)

    return power, fisher


def readfile_calcpsfisher(X):
    fname, nk, nz = X
    chunks = readOneFitsFile(fname)
    power, fisher = calc_total_ps_fisher(chunks, nk, nz)
    return chunks, power, fisher


def one_jackknife_est(X):
    chunk, total_power, total_fisher, di = X
    xpower = total_power.copy()
    xfisher = total_fisher.copy()

    xpower = chunk.add_to_total_power(xpower, m=-1)
    xfisher = chunk.add_to_total_fisher(xfisher, m=-1)
    xfisher[di] = np.where(xfisher[di] == 0, 1, xfisher[di])
    c, low = cho_factor(xfisher)
    xpower = cho_solve((c, low), xpower)
    return xpower


def block_jackknife_est(X):
    chunks, total_power, total_fisher, di = X
    xpower = total_power.copy()
    xfisher = total_fisher.copy()

    for chunk in chunks:
        xpower = chunk.add_to_total_power(xpower, m=-1)
        xfisher = chunk.add_to_total_fisher(xfisher, m=-1)
    xfisher[di] = np.where(xfisher[di] == 0, 1, xfisher[di])
    c, low = cho_factor(xfisher)
    xpower = cho_solve((c, low), xpower)
    return xpower


def main():
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "BootChunkFileBase", help="BootChunkFile as described in QMLE.")
    parser.add_argument("Nk", help="Number of k bins", type=int)
    parser.add_argument("Nz", help="Number of z bins", type=int)
    parser.add_argument(
        "--nblocks", help="Use block jackknife instead.", type=int)
    # parser.add_argument("--save-powers", action="store_true")
    # parser.add_argument("--calculate-originals", action="store_true")
    parser.add_argument("--fbase", default="")
    parser.add_argument("--nproc", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    outdir = ospath_dir(args.BootChunkFileBase)
    if outdir == "":
        outdir = "."
    if args.fbase and args.fbase[-1] != '-':
        args.fbase += '-'

    all_bootfilenames = glob.glob(f"{args.BootChunkFileBase}-*.fits")
    if len(all_bootfilenames) == 0:
        raise RuntimeError("Boot chunk files not found.")

    args_list = [(fname, args.Nk, args.Nz) for fname in all_bootfilenames]
    all_chunks = []
    ntot = args.Nk * args.Nz
    total_fisher = np.zeros((ntot, ntot))
    total_power = np.zeros(ntot)

    pool = Pool(processes=args.nproc)
    imap_it = pool.imap(readfile_calcpsfisher, args_list)

    logging.info("Reading chunk files and calculating total power & fisher...")
    for chunks, power, fisher in tqdm(imap_it):
        all_chunks.extend(chunks)
        total_power += power
        total_fisher += fisher

    nchunks = len(all_chunks)
    if args.nblocks and args.nblocks > 1 and args.nblocks < nchunks / 10:
        indices = np.linspace(0, nchunks, args.nblocks + 1).astype(int)
        all_chunks = [
            all_chunks[indices[_]:indices[_ + 1]]
            for _ in range(args.nblocks)
        ]

        jackknife_method = block_jackknife_est
    else:
        jackknife_method = one_jackknife_est

    logging.info("Calculating all jackknife estimates...")
    di = np.diag_indices(total_power.size)
    args_list = [
        (chunk, total_power, total_fisher, di)
        for chunk in all_chunks
    ]
    all_powers = []
    imap_it = pool.imap(jackknife_method, args_list)
    for xpower in tqdm(imap_it):
        all_powers.append(xpower)

    logging.info("Done.")
    pool.close()

    logging.info("Calculating jackknife covariance...")
    all_powers = np.vstack(all_powers)
    jackknife_cov = np.cov(all_powers, rowvar=False)
    output_fname = ospath_join(outdir, f"{args.fbase}jackknife-chunks-cov.txt")
    np.savetxt(output_fname, jackknife_cov)
    logging.info(f"Covariance saved as {output_fname}.")
