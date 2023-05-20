import argparse
import cProfile
import glob
from multiprocessing import Pool, RawArray
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


def get_parser():
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
    parser.add_argument(
        "--profile", type=int, default=None,
        help="Pass integer to limit number chunks and enable profiling.")
    return parser


# Global total power and fisher
g_total_power = None
g_total_fisher = None
g_dia_indices = None
g_size = None


def init_worker(tp, tf, s):
    global g_total_power
    global g_total_fisher
    global g_dia_indices
    global g_size

    g_total_power = tp
    g_total_fisher = tf
    g_dia_indices = np.diag_indices(s)
    g_size = s


def my_cho_solve(fisher, power):
    di = g_dia_indices
    fisher[di] = np.where(np.isclose(fisher[di], 0), 1, fisher[di])
    return cho_solve(cho_factor(fisher), power)


def one_jackknife_est(chunk):
    xpower = np.frombuffer(g_total_power).copy()
    xfisher = np.frombuffer(g_total_fisher).reshape((g_size, g_size)).copy()

    xpower = chunk.add_to_total_power(xpower, m=-1)
    xfisher = chunk.add_to_total_fisher(xfisher, m=-1)

    return my_cho_solve(xfisher, xpower)


def block_jackknife_est(chunks):
    xpower = np.frombuffer(g_total_power).copy()
    xfisher = np.frombuffer(g_total_fisher).reshape((g_size, g_size)).copy()

    for chunk in chunks:
        xpower = chunk.add_to_total_power(xpower, m=-1)
        xfisher = chunk.add_to_total_fisher(xfisher, m=-1)

    return my_cho_solve(xfisher, xpower)


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

    for hdu in f:
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


def read_set_totals(all_bootfilenames, nk, nz, nproc):
    args_list = [(fname, nk, nz) for fname in all_bootfilenames]
    all_chunks = []
    ntot = nk * nz
    total_fisher = np.zeros((ntot, ntot))
    total_power = np.zeros(ntot)

    pool = Pool(processes=nproc)
    imap_it = pool.imap(readfile_calcpsfisher, args_list)

    logging.info("Reading chunk files and calculating total power & fisher...")
    for chunks, power, fisher in tqdm(imap_it, total=len(args_list)):
        all_chunks.extend(chunks)
        total_power += power
        total_fisher += fisher
    pool.close()

    return all_chunks, total_power, total_fisher


def get_jackknife_method(all_chunks, nblocks):
    nchunks = len(all_chunks)
    if nblocks and nblocks > 1 and nblocks < nchunks / 10:
        logging.info(f"Using {nblocks} blocks for jackknife.")
        indices = np.linspace(0, nchunks, nblocks + 1).astype(int)
        all_chunks = [
            all_chunks[indices[_]:indices[_ + 1]]
            for _ in range(nblocks)
        ]

        jackknife_method = block_jackknife_est
    else:
        jackknife_method = one_jackknife_est

    return all_chunks, jackknife_method


def calc_all_jackknife_estimates(
        all_chunks, jackknife_method, total_power, total_fisher, nproc
):
    logging.info("Calculating all jackknife estimates...")
    ntot = total_power.size

    r_tot_p = RawArray('d', ntot)
    np_tot_p = np.frombuffer(r_tot_p)
    np.copyto(np_tot_p, total_power)

    r_tot_f = RawArray('d', ntot * ntot)
    np_tot_f = np.frombuffer(r_tot_f).reshape(ntot, ntot)
    np.copyto(np_tot_f, total_fisher)

    pool = Pool(
        processes=nproc, initializer=init_worker,
        initargs=(r_tot_p, r_tot_f, ntot)
    )
    imap_it = pool.imap(jackknife_method, all_chunks)
    all_powers = np.empty((len(all_chunks), ntot))
    jj = 0
    for xpower in tqdm(imap_it, total=len(all_chunks)):
        all_powers[jj] = xpower
        jj += 1

    logging.info("Done.")
    pool.close()

    return all_powers


def run(all_bootfilenames, outdir, args):
    all_chunks, total_power, total_fisher = \
        read_set_totals(all_bootfilenames, args.Nk, args.Nz, args.nproc)

    global g_dia_indices
    g_dia_indices = np.diag_indices(total_power.size)
    logging.info(f"Original result {my_cho_solve(total_fisher, total_power)}")

    if args.profile and args.profile > 0:
        all_chunks = all_chunks[:args.profile]

    all_chunks, jackknife_method = \
        get_jackknife_method(all_chunks, args.nblocks)

    all_powers = calc_all_jackknife_estimates(
        all_chunks, jackknife_method, total_power, total_fisher, args.nproc)

    logging.info("Calculating jackknife covariance...")
    jackknife_cov = np.cov(all_powers, rowvar=False)
    output_fname = ospath_join(outdir, f"{args.fbase}jackknife-chunks-cov.txt")
    np.savetxt(output_fname, jackknife_cov)
    logging.info(f"Covariance saved as {output_fname}.")


def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG)

    outdir = ospath_dir(args.BootChunkFileBase)
    if outdir == "":
        outdir = "."
    if args.fbase and args.fbase[-1] != '-':
        args.fbase += '-'

    all_bootfilenames = glob.glob(f"{args.BootChunkFileBase}-*.fits")
    if len(all_bootfilenames) == 0:
        raise RuntimeError("Boot chunk files not found.")

    if args.profile and args.profile > 0:
        pr = cProfile.Profile()
        pr.enable()

    run(all_bootfilenames, outdir, args)

    if args.profile and args.profile > 0:
        pr.disable()
        pr.print_stats(sort='cumulative')
