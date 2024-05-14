import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from os.path import dirname as ospath_dir


def normalize(matrix, return_vector=False):
    fk_v = matrix.diagonal().copy()
    w = fk_v == 0
    fk_v[w] = 1
    fk_v = np.sqrt(fk_v)

    R = matrix / np.outer(fk_v, fk_v)
    R[w, :] = 0
    R[:, w] = 0
    if return_vector:
        fk_v[w] = 0
        return R, fk_v
    return R


def replace_zero_diags(matrix):
    newmatrix = matrix.copy()
    di = np.diag_indices(matrix.shape[0])
    w = matrix[di] <= 0
    newmatrix[di] = np.where(w, 1, matrix[di])
    return newmatrix, w, di


def safe_inverse(matrix):
    invmatrix = matrix.copy()

    zero_diag_idx = np.where(invmatrix.diagonal() == 0)[0]
    for idx in zero_diag_idx:
        invmatrix[idx, idx] = 1

    return np.linalg.inv(invmatrix), zero_diag_idx

# evecs.T @ cov_boot @ evecs
# Pat actually uses SVD, but that doesn't ensure positive definiteness
# He also uses bootstrap evecs
# Here, I tried to do the same with eigenvalues.
# This way positive definiteness is satisfied.
# My findings: Bootstrap evecs works, chi2 slightly above mean dof
#              QMLE evecs also works, chi2 slightly below mean dof


def smooth_matrix(boot_mat, qmle_mat, nz, reg_in_cov, sigma=2.0):
    rboot, vboot = normalize(boot_mat, return_vector=True)
    rqmle, vqmle = normalize(qmle_mat, return_vector=True)
    rnew = normalize(
        gaussian_filter(rboot - rqmle, sigma) + rqmle
    )
    rnew += rnew.T
    rnew /= 2

    if reg_in_cov:
        v = np.fmax(vboot, vqmle)
    else:
        v = np.fmin(vboot, vqmle)

    w = vqmle == 0
    vqmle[w] = 1
    eta = np.reshape(v / vqmle - 1, nz, vqmle.size // nz)
    eta = gaussian_filter1d(eta, sigma=0.5, axis=0)
    for i in range(eta.shape[0]):
        eta[i] = gaussian_filter1d(eta[i], sigma=eta[i].std() / 0.025)
    v = (1 + eta).ravel() * vqmle
    v[w] = 0

    return np.outer(v, v) * rnew


def mcdonald_eval_fix(boot_mat, qmle_mat, reg_in_cov):
    boot_mat = replace_zero_diags(boot_mat)[0]
    qmle_mat, wzero, _ = replace_zero_diags(qmle_mat)

    evals, evecs = np.linalg.eigh(boot_mat)

    other_svals = np.array([v.dot(qmle_mat.dot(v)) for v in evecs.T])

    if reg_in_cov:
        w = evals < other_svals
    else:
        w = evals > other_svals

    print(f"Changed {w.sum()} modes.")

    evals[w] = other_svals[w]
    newmatrix = evecs @ np.diag(evals) @ evecs.T

    newmatrix[wzero, :] = 0
    newmatrix[:, wzero] = 0

    return newmatrix


def main():
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--boot-matrix", help="Covariance or Fisher from bootstrap file.",
        required=True)
    parser.add_argument(
        "--qmle-fisher", help="Fisher matrix from QMLE.", required=True)
    parser.add_argument(
        "--qmle-cov", help="Covariance matrix from QMLE.", required=True)
    parser.add_argument(
        "--nz", help="Number of redshift bins", required=True)
    parser.add_argument(
        "--qmle-sparcity-cut", default=0.001, type=float,
        help="Sparsity pattern to cut using QMLE Fisher matrix.")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of iterations")
    parser.add_argument("--reg-in-cov", action="store_true")
    parser.add_argument("--fbase", default="")
    args = parser.parse_args()

    outdir = ospath_dir(args.boot_matrix)
    bootstrap_matrix = np.loadtxt(args.boot_matrix, skiprows=1)
    assert bootstrap_matrix.shape[0] % args.nz == 0

    qmle_fisher = np.loadtxt(args.qmle_fisher, skiprows=1)
    qmle_zero_idx = np.where(qmle_fisher.diagonal() == 0)[0]
    qmle_covariance = np.loadtxt(args.qmle_cov, skiprows=1)

    normalized_qmle_cov = normalize(qmle_covariance)
    normalized_qmle_fisher = normalize(qmle_fisher)

    matrix_to_use_for_sparsity = (
        normalized_qmle_cov if args.reg_in_cov else normalized_qmle_fisher)
    matrix_to_use_for_input_qmle = (
        qmle_covariance if args.reg_in_cov else qmle_fisher)

    bootstrap_matrix = smooth_matrix(
        bootstrap_matrix, matrix_to_use_for_input_qmle, args.reg_in_cov)
    # prevent leakage to zero elements
    bootstrap_matrix[qmle_zero_idx, :] = 0
    bootstrap_matrix[:, qmle_zero_idx] = 0

    if args.qmle_sparcity_cut > 0:
        qmle_sparcity = np.abs(
            matrix_to_use_for_sparsity) > args.qmle_sparcity_cut
    else:
        qmle_sparcity = np.ones(qmle_fisher.shape, dtype=bool)

    for it in range(args.iterations):
        print(f"Iteration {it}.")

        newmatrix = mcdonald_eval_fix(
            bootstrap_matrix * qmle_sparcity,
            matrix_to_use_for_input_qmle, args.reg_in_cov)

        if np.allclose(0, bootstrap_matrix - newmatrix):
            print("Converged.")
            bootstrap_matrix = newmatrix
            break

        bootstrap_matrix = newmatrix

    covfis_txt = "cov" if args.reg_in_cov else "fisher"
    finalfname = (
        f"{args.fbase}regularized-bootstrap-{covfis_txt}-"
        f"s{args.qmle_sparcity_cut:.3f}-boot-evecs.txt")
    np.savetxt(
        f"{outdir}/{finalfname}", bootstrap_matrix,
        header=f"{qmle_covariance.shape[0]} {qmle_covariance.shape[0]}")
