import argparse
import numpy as np
from os.path import dirname as ospath_dir


def normalize(matrix):
    fk_v = matrix.diagonal()
    norm = np.sqrt(np.outer(fk_v, fk_v))
    norm[norm == 0] = 1
    return matrix / norm


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


def mcdonald_eval_fix(boot_mat, qmle_mat, reg_in_cov):
    evals, evecs = np.linalg.eigh(boot_mat)

    other_svals = np.array([v.dot(qmle_mat.dot(v)) for v in evecs.T])

    if reg_in_cov:
        w = evals < other_svals
    else:
        w = evals > other_svals

    evals[w] = other_svals[w]
    newmatrix = evecs @ np.diag(evals) @ evecs.T

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
        "--qmle-sparcity-cut", default=0.001, type=float,
        help="Sparsity pattern to cut using QMLE Fisher matrix.")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of iterations")
    parser.add_argument("--reg-in-cov", action="store_true")
    parser.add_argument("--fbase", default="")
    args = parser.parse_args()

    outdir = ospath_dir(args.boot_cov)
    bootstrap_matrix = np.loadtxt(args.boot_matrix, skiprows=1)

    qmle_fisher = np.loadtxt(args.qmle_fisher, skiprows=1)
    qmle_zero_idx = np.where(qmle_fisher.diagonal() == 0)[0]
    qmle_covariance = np.loadtxt(args.qmle_cov, skiprows=1)

    normalized_qmle_cov = normalize(qmle_covariance)
    normalized_qmle_fisher = normalize(qmle_fisher)

    matrix_to_use_for_sparsity = (
        normalized_qmle_cov if args.reg_in_cov else normalized_qmle_fisher)
    matrix_to_use_for_input_qmle = (
        qmle_covariance if args.reg_in_cov else qmle_fisher)

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

        if np.allclose(bootstrap_matrix, newmatrix):
            print("Converged.")
            bootstrap_matrix = newmatrix
            break

        bootstrap_matrix = newmatrix

    for idx in qmle_zero_idx:
        bootstrap_matrix[idx, :] = 0
        bootstrap_matrix[:, idx] = 0

    covfis_txt = "cov" if args.reg_in_cov else "fisher"
    finalfname = (
        f"{args.fbase}regularized-bootstrap-{covfis_txt}-"
        f"s{args.qmle_sparcity_cut:4f}-boot-evecs.txt")
    np.savetxt(
        f"{outdir}/{finalfname}", bootstrap_matrix,
        header=f"{qmle_covariance.shape[0]} {qmle_covariance.shape[0]}")
