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


def mcdonald_eval_fix(boot_cov, qmle_cov, use_boot_base_evecs=True):
    if use_boot_base_evecs:
        # 1) Use bootstrap evecs
        evals_boot, evecs_boot = np.linalg.eigh(boot_cov)

        # s_cov_qmle_bootevecs = np.diag(evecs_boot.T @ qmle_cov @ evecs_boot)
        s_cov_qmle_bootevecs = np.array(
            [v.T.dot(qmle_cov.dot(v)) for v in evecs_boot.T])

        small_vals = evals_boot < s_cov_qmle_bootevecs
        evals_boot[small_vals] = s_cov_qmle_bootevecs[small_vals]
        newcov = evecs_boot @ np.diag(evals_boot) @ evecs_boot.T
    else:
        # 2) Use QMLE evecs
        evals_qmle, evecs_qmle = np.linalg.eigh(qmle_cov)

        s_cov_boot_qmleevecs = np.diag(evecs_qmle.T @ boot_cov @ evecs_qmle)

        small_vals = evals_qmle < s_cov_boot_qmleevecs
        evals_qmle[small_vals] = s_cov_boot_qmleevecs[small_vals]
        newcov = evecs_qmle @ np.diag(evals_qmle) @ evecs_qmle.T

    return newcov


def main():
    # Arguments passed to run the script
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--boot-cov", help="Covariance from bootstrap file.", required=True)
    parser.add_argument(
        "--qmle-fisher", help="Fisher matrix from QMLE.", required=True)
    parser.add_argument(
        "--qmle-cov", help="Covariance matrix from QMLE.", required=True)
    parser.add_argument(
        "--qmle-sparcity-cut", default=0.001, type=float,
        help="Sparsity pattern to cut using QMLE Fisher matrix.")
    parser.add_argument("--use-qmle-evecs", action="store_true")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of iterations")
    parser.add_argument("--reg-in-cov", action="store_true")
    parser.add_argument("--fbase", default="")
    args = parser.parse_args()

    outdir = ospath_dir(args.boot_cov)
    bootstrap_covariance = np.loadtxt(args.boot_cov)

    qmle_fisher = np.loadtxt(args.qmle_fisher, skiprows=1)
    qmle_zero_idx = np.where(qmle_fisher.diagonal() == 0)[0]
    qmle_covariance = np.loadtxt(args.qmle_cov, skiprows=1)

    normalized_qmle_cov = normalize(qmle_covariance)
    normalized_qmle_fisher = normalize(qmle_fisher)

    matrix_to_use_for_sparsity = (
        normalized_qmle_cov if args.reg_in_cov else normalized_qmle_fisher)

    if args.qmle_sparcity_cut > 0:
        qmle_sparcity = np.abs(
            matrix_to_use_for_sparsity) > args.qmle_sparcity_cut
    else:
        qmle_sparcity = np.ones(qmle_fisher.shape, dtype=bool)

    for it in range(args.iterations):
        print(f"Iteration {it}.")
        if args.reg_in_cov:
            newcov = bootstrap_covariance * qmle_sparcity
        else:
            boostrap_fisher, _ = safe_inverse(bootstrap_covariance)
            newcov, _ = safe_inverse(boostrap_fisher * qmle_sparcity)

        newcov = mcdonald_eval_fix(
            newcov, qmle_covariance, not args.use_qmle_evecs)
        diff = np.abs(newcov - bootstrap_covariance)
        bootstrap_covariance = newcov

        if np.all(diff < 1e-8):
            print("Converged.")
            break

    for idx in qmle_zero_idx:
        bootstrap_covariance[idx, :] = 0
        bootstrap_covariance[:, idx] = 0

    basis_txt = "qmle" if args.use_qmle_evecs else "boot"
    finalfname = (
        f"{args.fbase}regularized-bootstrap-cov-"
        f"s{args.qmle_sparcity_cut:4f}-{basis_txt}-evecs.txt")
    np.savetxt(
        f"{outdir}/{finalfname}", bootstrap_covariance,
        header=f"{qmle_covariance.shape[0]} {qmle_covariance.shape[0]}")
