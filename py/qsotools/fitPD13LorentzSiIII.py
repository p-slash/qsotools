import argparse

import numpy as np
from scipy.optimize import minimize
from astropy.io import ascii

import qsotools.fiducial as fid

ions_modeled = ['Si II', 'Si III']
velocity_seps = {'Si II': 5577., 'Si III': 2271.}

class LyaP1DModel(object):
    def __init__(self, fname_qmle, fname_cov, ions):
        self.power_table = np.array(ascii.read(fname_qmle))
        self.cov = np.loadtxt(fname_cov)
        self.fit_ions = ions

        self.names = ['A', 'n', 'alpha', 'B', 'beta', 'k1']
        self.initial = {
            'A':PDW_FIT_AMP,
            'n':PDW_FIT_N,
            'alpha':PDW_FIT_APH,
            'B':PDW_FIT_B,
            'beta':PDW_FIT_BETA,
            'k1':1/np.sqrt(PDW_FIT_LMD)
        }

        for ion in self.fit_ions:
            self.names.append(f'f_{ion}')
            self.initial[f'f_{ion}']=0.


    def getModelPower(self, **kwargs):
        lmd = 1./kwargs['k1']**2
        X = self.power_table['kc'], self.power_table['z']
        plya = fid.evaluatePD13Lorentz(X, kwargs['A'], kwargs['n'],
            kwargs['alpha'], kwargs['B'], kwargs['beta'],
            lmd)
        for ion in self.fit_ions:
            f = kwargs[f'f_{ion}']
            plya *= (1+f**2+2*f*np.cos(self.power_table['kc']*velocity_seps[ion]))

        return plya

    def chi2(self, *args):
        kwargs = {par: args[i] for i, par in enumerate(self.names)}
        pmodel = self.getModelPower(**kwargs)
        diff = pmodel - self.power_table['p_final']
        return diff@np.linalg.inv(self.cov)@diff

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--final-qmle", help="Final results file", required=True)
    parser.add_argument("--final-cov", help="Final covariance with syst.", required=True)
    parser.add_argument("--ions", help="Si ions to fit.", nargs='*', default=ions_modeled)
    args = parser.parse_args()

    model = LyaP1DModel(args.final_qmle, args.final_cov, args.ions)

    result = minimize(model.chi2, model.initial)
    chi2 = model.chi2(result.x)
    print(f"Chi2: {chi2} / {len(model.names)}")

    for i, name in enumerate(model.names):
        print(f"{name}: {results.x[i]} pm {result.hess_inv[i, i]}")

    print("Full inv hess")
    print(result.hess_inv)



