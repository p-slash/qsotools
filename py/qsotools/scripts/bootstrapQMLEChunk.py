import cProfile
import glob
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

import qsotools.scripts.bootstrapQMLE as qsoboot


class Chunk():
    @classmethod
    def list_from_fitsfile(cls, fname):
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

    def __init__(self, data, ndim, istart):
        self.ndim = ndim
        self.istart = istart
        self.data = np.empty(data.size - 2 * ndim)
        self.data[:ndim] = (
            data[:ndim] - data[ndim:2 * ndim] - data[2 * ndim:3 * ndim])
        self.data[ndim:] = data[3 * ndim:]
        self.pk = self.data[:ndim]
        self.upper_fisher = self.data[ndim:]
        # self.fisher = _fast_construct_fisher(ndim, data[3 * ndim:])
        self.s1 = np.s_[istart:istart + ndim]


def get_parser():
    parser = qsoboot.get_parser()
    parser.add_argument("Nk", help="Number of k bins", type=int)
    parser.add_argument("Nz", help="Number of z bins", type=int)
    # parser.add_argument(
    #     "--nblocks", help="Use block bootstrap instead.", type=int)
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable profiling.")
    return parser


def my_cho_solve(fisher, power):
    di = np.diag_indices(power.size)
    fisher[di] = np.where(np.isclose(fisher[di], 0), 1, fisher[di])
    return cho_solve(cho_factor(fisher), power)


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


@njit("f8[:, :](f8[:, :], f8[:], i8, i8)")
def add_to_fisher(fisher, upper_fisher, ndim, istart):
    fari = 0
    for did in range(ndim):
        for jj in range(ndim - did):
            ii = jj + istart
            fisher[ii, ii + did] += upper_fisher[fari + jj]
            fisher[ii + did, ii] += upper_fisher[fari + jj]

        fari += ndim - did

    return fisher


def calc_total_ps_fisher(chunks, nk, nz):
    ntot = nk * nz
    fisher = np.zeros((ntot, ntot))
    power = np.zeros(ntot)

    for chunk in tqdm(chunks):
        power[chunk.s1] += chunk.pk
        fisher = add_to_fisher(
            fisher, chunk.upper_fisher, chunk.ndim, chunk.istart)

    return power, fisher


def read_all_chunks(all_bootfilenames):
    all_chunks = []
    imap_it = (Chunk.list_from_fitsfile(fname) for fname in all_bootfilenames)

    logging.info("Reading chunk files and calculating total power & fisher...")
    for chunks in tqdm(imap_it, total=len(all_bootfilenames)):
        all_chunks.extend(chunks)

    return all_chunks


# def get_jackknife_method(all_chunks, nblocks):
#     nchunks = len(all_chunks)
#     if nblocks and nblocks > 1 and nblocks < nchunks / 10:
#         logging.info(f"Using {nblocks} blocks for jackknife.")
#         indices = np.linspace(0, nchunks, nblocks + 1).astype(int)
#         all_chunks = [
#             all_chunks[indices[_]:indices[_ + 1]]
#             for _ in range(nblocks)
#         ]

#         jackknife_method = block_jackknife_est
#     else:
#         jackknife_method = one_jackknife_est

#     return all_chunks, jackknife_method


def getOneSliceBoot(
        spectra, booted_indices, nspec,
        Nk, Nd, total_nkz, elems_count,
        remove_last_nz_bins, nboot_per_it
):
    boot_counts = qsoboot.getCounts(booted_indices)
    logging.info("    > Generated boot indices.")

    # Allocate memory for matrices
    # total_data = np.empty((nboot_per_it, elems_count))
    total_data = boot_counts @ spectra

    logging.info("    > Calculating bootstrapped inverse Fisher and power...")
    total_power_b4, F = getPSandFisher(
        total_data, Nk, Nd, total_nkz, remove_last_nz_bins)
    total_power = 0.5 * my_cho_solve(F, total_power_b4)

    return total_power


def calculate_original(all_chunks, outdir, args):
    if not args.calculate_originals:
        return

    logging.info("Calculating original power.")
    power, fisher = calc_total_ps_fisher(all_chunks, args.Nk, args.Nz)

    orig_power = my_cho_solve(fisher, power)

    # Save power to a file
    # Set up output file
    output_fname = ospath_join(
        outdir, f"{args.fbase}bootstrap-original-power.txt")
    np.savetxt(output_fname, orig_power)
    logging.info(f"Original power saved as {output_fname}.")

    output_fname = ospath_join(
        outdir, f"{args.fbase}bootstrap-original-fisher.txt")
    np.savetxt(output_fname, 0.5 * fisher)
    logging.info(f"Original fisher saved as {output_fname}.")


def run(all_bootfilenames, outdir, args):
    all_chunks = read_all_chunks(all_bootfilenames)
    total_nkz = args.Nk * args.Nz
    nspec = len(all_chunks)

    calculate_original(all_chunks, outdir, args)

    g_dia_indices = np.diag_indices(total_power.size)

    # all_chunks, jackknife_method = \
    #     get_jackknife_method(all_chunks, args.nblocks)
    # all_powers = calc_all_jackknife_estimates(
    #     all_chunks, jackknife_method, total_power, total_fisher, args.nproc)

    # logging.info("Calculating jackknife covariance...")
    # jackknife_cov = np.cov(all_powers, rowvar=False)
    # output_fname = ospath_join(outdir, f"{args.fbase}jackknife-chunks-cov.txt")
    # np.savetxt(output_fname, jackknife_cov)
    # logging.info(f"Covariance saved as {output_fname}.")

    # Generate bootstrap realizations through indexes
    RND = np.random.default_rng(args.seed)
    newpowersize = total_nkz - args.remove_last_nz_bins * args.Nk
    total_power = np.empty((args.bootnum, newpowersize))
    n_iter = args.bootnum // args.nboot_per_it

    for jj in tqdm(range(n_iter)):
        logging.info(f"Iteration {jj+1}/{n_iter}.")
        i1 = jj * args.nboot_per_it
        i2 = i1 + args.nboot_per_it
        booted_indices = RND.integers(
            low=0, high=nspec, size=(args.nboot_per_it, nspec))
        total_power[i1:i2] = getOneSliceBoot(
            spectra, booted_indices, nspec,
            Nk, Nd, total_nkz, elems_count,
            args.remove_last_nz_bins, args.nboot_per_it)

    bootstrap_cov = np.cov(total_power, rowvar=False)
    output_fname = ospath_join(
        outdir, f"{args.fbase}bootstrap-cov-n{args.bootnum}-s{args.seed}.txt")
    np.savetxt(output_fname, bootstrap_cov)
    logging.info(f"Covariance saved as {output_fname}.")

    if args.save_powers:
        output_fname = ospath_join(
            outdir,
            f"{args.fbase}bootstrap-power-n{args.bootnum}-s{args.seed}.txt")
        np.savetxt(output_fname, total_power)
        logging.info(f"Power saved as {output_fname}.")


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

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    run(all_bootfilenames, outdir, args)

    if args.profile and args.profile > 0:
        pr.disable()
        pr.print_stats(sort='cumulative')
