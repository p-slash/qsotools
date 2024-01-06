import numpy as np
from scipy.interpolate import CubicSpline

import qsotools.fiducial as qfid


def findSvdJump(M, svd=None, jump=8.):
    if svd is None:
        svd = np.linalg.svd(M, compute_uv=False)
    ratios_svd = svd[:-1] / svd[1:]

    jj = 0
    for _ in ratios_svd[::-1]:
        if _ > jump:
            jj += 1
        else:
            break

    return jj, svd[-jj]


def invertDampSvd(fisher, jump=8.):
    jj, x = findSvdJump(fisher, None, jump)

    if jj == 0:
        return np.linalg.inv(fisher)
    print("Damp:", x)

    return invertSymDamped(fisher, x)


def invertSymDamped(S, damp):
    di = np.diag_indices(S.shape[0])
    newf = S.dot(S)
    newf[di] += damp

    inv = np.linalg.inv(newf).dot(S)
    return (inv + inv.T) / 2


def invertRegularizedCorrCoeff(S, target_cond=100.):
    V = np.sqrt(S.diagonal())
    V = np.outer(V, V)
    R = S / V
    eigvals = np.linalg.eigvalsh(R)
    delta = max(0, (eigvals[-1] - target_cond * eigvals[0]) / (target_cond - 1))

    if delta == 0:
        return np.linalg.inv(S)

    print("Damp:", delta)
#     di = np.diag_indices(S.shape[0])
#     R[di] += delta
#     R /= 1 + delta
#     R = np.linalg.inv(R)

    return invertSymDamped(R, delta) / V


class QmleSimulator():
    """docstring for QmleSimulator"""

    def __init__(
            self, zmed=2.4, dlambda=0.8, sigma=0.5, nklin=60, nklog=25,
            dklin=5e-4, dklog=0.01, nfft=2**20
    ):
        self.dlambda = dlambda
        self.noise_amp = sigma
        self.k_edges, self.k_centers = qfid.formBins(
            nklin, nklog, dklin, dklog, 0, klast=-1)
        self.nkbins = self.k_centers.size

        w1 = (1 + zmed - 0.1) * qfid.LYA_WAVELENGTH
        w2 = (1 + zmed + 0.1) * qfid.LYA_WAVELENGTH
        self.nwave = int((w2 - w1) / dlambda) + 1
        self.wave = np.linspace(w1, w1 + (self.nwave - 1) * dlambda, self.nwave)

        self.z = self.wave.mean() / qfid.LYA_WAVELENGTH - 1
        self.dv = qfid.LIGHT_SPEED * dlambda / (1 + self.z) / qfid.LYA_WAVELENGTH
        self.R_kms = 0.8 * self.dv

        self.varr = qfid.LIGHT_SPEED * np.log(self.wave / qfid.LYA_WAVELENGTH)
        self.dv_matrix = np.abs(self.varr[:, np.newaxis] - self.varr[np.newaxis, :])

        self.nfft = nfft
        self.vfft = np.arange(self.nfft // 2) * 10.
        self.kfft = 2. * np.pi * np.fft.rfftfreq(self.nfft, d=10.)

        self.sfid_mat = None
        self.noise_vec = None
        self.cov_mat = None
        self.inv_cov_mat = None
        self.qk_matrices = None

        self.dk = None
        self.bk = None
        self.fisher = None
        self.invfisher = None

        self._random_mask_idx = None

    def setRandomMasking(self, npix):
        if npix <= 0:
            self._random_mask_idx = None

        if (npix > self.nwave / 2):
            print("Warning masking more than half")

        self._random_mask_idx = np.random.default_rng().choice(
            self.nwave, size=npix, replace=False)
        self._random_mask_idx.sort()

    def getWindowFncK(self):
        kR2 = self.kfft**2 * self.R_kms**2
        return np.sinc(self.kfft * self.dv / 2 / np.pi)**2 * np.exp(-kR2)

    def getSfidMatrix(self):
        pfid = qfid.evaluatePD13Lorentz(
            (self.kfft, self.z), *qfid.DESI_EDR_PARAMETERS)
        pfid *= self.getWindowFncK()
        pfid[0] = 0
        xi = np.fft.irfft(pfid)[:self.nfft // 2] / self.dv

        self.sfid_mat = CubicSpline(self.vfft, xi)(self.dv_matrix)

        return self.sfid_mat

    def getCovarianceMatrix(self):
        if self.sfid_mat is None:
            self.getSfidMatrix()

        self.cov_mat = self.sfid_mat.copy()

        self.noise_vec = np.ones(self.nwave) * self.noise_amp**2
        if self._random_mask_idx is not None:
            self.noise_vec[self._random_mask_idx] = 1e16

        self.cov_mat[np.diag_indices(self.nwave)] += self.noise_vec

        return self.cov_mat

    def getInverseCovarianceMatrix(self, cont_order=-1):
        self.inv_cov_mat = np.linalg.inv(self.getCovarianceMatrix())

        if cont_order < 0:
            return self.inv_cov_mat

        template_matrix = np.vander(
            np.log(self.wave / qfid.LYA_WAVELENGTH), cont_order + 1)
        U, s, _ = np.linalg.svd(template_matrix, full_matrices=False)

        # Remove small singular valued vectors
        w = s > 1e-6
        U = U[:, w]  # shape = (self.size, cont_order + 1)
        Y = self.inv_cov_mat @ U
        # Woodbury formula. Note that U and Y are not square matrices.
        self.inv_cov_mat -= Y @ np.linalg.inv(U.T @ Y) @ Y.T

        return self.inv_cov_mat

    def getQkMatrices(self):
        self.qk_matrices = []
        window = self.getWindowFncK()
        for k in range(self.nkbins):
            qkw = np.zeros(self.kfft.size)
            i1, i2 = np.searchsorted(
                self.kfft, [self.k_edges[k], self.k_edges[k + 1]])
            qkw[i1:i2] = window[i1:i2]
            qkw = np.fft.irfft(qkw)[:self.nfft // 2] / self.dv

            self.qk_matrices.append(
                CubicSpline(self.vfft, qkw)(self.dv_matrix))

        return self.qk_matrices

    def simulate(self, cont_order=-1, random_masking_npix=0, nqso=1):
        self.sfid_mat = self.getSfidMatrix()
        qk_matrices = self.getQkMatrices()

        self.fisher = np.zeros((self.nkbins, self.nkbins))
        self.dk = np.zeros(self.nkbins)
        self.bk = np.zeros(self.nkbins)

        for _ in range(nqso):
            self.setRandomMasking(random_masking_npix)
            self.inv_cov_mat = self.getInverseCovarianceMatrix(cont_order)

            # Weight
            wqk_matrices = [self.inv_cov_mat.dot(_) for _ in qk_matrices]

            # Fisher matrix calculation
            for ki in range(self.nkbins):
                self.fisher[ki, ki:] += np.fromiter(
                    (np.vdot(wqk_matrices[ki], _.T) for _ in wqk_matrices[ki:]),
                    dtype=float,
                    count=self.nkbins - ki)

            # Estimate dk
            weighted_sfid = self.sfid_mat.dot(self.inv_cov_mat)
            self.dk += np.fromiter(
                (np.vdot(_, weighted_sfid) for _ in wqk_matrices),
                dtype=float,
                count=self.nkbins)

            Noise = (self.inv_cov_mat * self.noise_vec).T
            self.bk += np.fromiter(
                (np.vdot(_, Noise) for _ in wqk_matrices),
                dtype=float,
                count=self.nkbins)

        self.fisher += self.fisher.T
        self.fisher[np.diag_indices(self.nkbins)] /= 2

        self.invfisher = invertDampSvd(self.fisher)
        self.dk = self.invfisher.dot(self.dk)
        self.bk = self.invfisher.dot(self.bk)

        return self.dk, self.bk, self.fisher
