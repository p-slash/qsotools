import argparse

import numpy as np
import iminuit
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
            'A':fid.PDW_FIT_AMP,
            'n':fid.PDW_FIT_N,
            'alpha':fid.PDW_FIT_APH,
            'B':fid.PDW_FIT_B,
            'beta':fid.PDW_FIT_BETA,
            'k1':1/np.sqrt(fid.PDW_FIT_LMD)
        }

        for ion in self.fit_ions:
            self.names.append(f'a_{ion}')
            self.initial[f'a_{ion}']=0.


    def getModelPower(self, **kwargs):
        lmd = 1./kwargs['k1']**2
        X = self.power_table['kc'], self.power_table['z']
        plya = fid.evaluatePD13Lorentz(X, kwargs['A'], kwargs['n'],
            kwargs['alpha'], kwargs['B'], kwargs['beta'],
            lmd)
        for ion in self.fit_ions:
            a = kwargs[f'a_{ion}'] # /(1-fid.meanFluxFG08(self.power_table['z']))
            plya *= (1+a**2+2*a*np.cos(self.power_table['kc']*velocity_seps[ion]))

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
    mini = iminuit.Minuit(model.chi2, name=model.names, **model.initial)
    mini.errordef = 1
    mini.print_level = 1
    print(mini.migrad())

    if not mini.valid:
        print("Error: function minimum is not valid.")

    result = mini.values.to_dict()
    errors = mini.errors.to_dict()

    chi2 = model.chi2(*result.values())
    print(f"Chi2: {chi2:.2f} / {model.power_table.size-len(model.names)}")

    for i, name in enumerate(model.names):
        print(f"{name}: {result[name]:.3e} pm {errors[name]:.3e}")

    # print("Full inv hess")
    # print(result.hess_inv)




