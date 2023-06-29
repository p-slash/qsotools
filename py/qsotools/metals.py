from collections import namedtuple
from itertools import combinations

import numpy as np
from scipy.special import voigt_profile as Voigt1D
from scipy.optimize import curve_fit
from scipy.integrate import quad

import qsotools.fiducial as fid

Transition = namedtuple('Transition', ['wave', 'f_osc'])

IonTransitions = { 
'C IV': [Transition(1550.77, 0.095), Transition(1548.20, 0.190)],
'Mg II': [Transition(2803.53, 0.314), Transition(2796.35, 0.629)],
'Si IV': [Transition(1402.77, 0.260), Transition(1393.75, 0.524)],
'H I': [Transition(1215.67, 0.416)]
}

Ions = list(IonTransitions.keys())

def getIonVelocitySeparations(ion):
    tpairs = combinations(IonTransitions[ion], 2)
    vseps = []
    for pair in tpairs:
        vseps.append(fid.LIGHT_SPEED*np.abs(np.log(pair[1].wave/pair[0].wave)))
    return {ion: vseps}

def getAllIonVelocitySeparations():
    result = {}
    for ion in IonTransitions:
        result.update( getIonVelocitySeparations(ion) )
    return result

class AbsorptionProfile(object):
    qe = 4.803204e-10 # statC, cm^3/2 g^1/2 s^-1
    me = 9.109384e-28 # g
    c_cms = fid.LIGHT_SPEED * 1e5 # km to cm for c
    aij_coeff = np.pi * qe**2 / me / c_cms # cm^2 s^-1

    # There could be a missing 2 compared to Tie 2022
    # def _getOpticalDepth(nu, lambda12_A, log10N, b, f12):
    #     a12 = AbsorptionProfile.aij_coeff * f12
    #     nu12 = fid.LIGHT_SPEED*1e13/lambda12_A
    #     gamma = 2 * nu12**2 / AbsorptionProfile.c_cms**2 * a12
        
    #     sigma_gauss = nu12 * b / fid.LIGHT_SPEED / np.sqrt(2)
    #     vfnc = Voigt1D(nu-nu12, sigma_gauss, gamma)

    #     tau = 10**log10N * (a12/nu12) * vfnc * nu

    #     return tau
    # def getFlux(wave, ion, log10N, b):
    #     nu = fid.LIGHT_SPEED * 1e13 / wave

    #     if isinstance(wave, np.ndarray):
    #         tau = np.zeros_like(wave)
    #     else:
    #         tau = 0

    #     for transition in IonTransitions[ion]:
    #         tau += AbsorptionProfile._getOpticalDepth(nu, transition.wave, log10N, b, transition.f_osc)

    #     return np.exp(-tau)

    def _getOpticalDepth(wave_A, lambda12_A, log10N, b, f12):
        a12 = AbsorptionProfile.aij_coeff * f12
        gamma = (2e3 / fid.LIGHT_SPEED) * a12 / lambda12_A
        
        sigma_gauss = b / fid.LIGHT_SPEED / np.sqrt(2)
        vfnc = Voigt1D(lambda12_A/wave_A - 1, sigma_gauss, gamma)

        tau = a12 * 10**(log10N-13) * (lambda12_A/fid.LIGHT_SPEED) * vfnc * (lambda12_A/wave_A)

        return tau

    def getFlux(wave, ion, log10N, b):
        if isinstance(wave, np.ndarray):
            tau = np.zeros_like(wave)
        else:
            tau = 0

        for transition in IonTransitions[ion]:
            tau += AbsorptionProfile._getOpticalDepth(wave, transition.wave, log10N, b, transition.f_osc)

        return np.exp(-tau)
    
    # Returns amplitude (double) and relative strengths (np.ndarray)
    def getRelativeFluxStrengths(ion, log10N, b):
        transitions = IonTransitions[ion]
        ntransitions = len(transitions)

        maximafluxes = np.zeros(ntransitions)

        for i in range(ntransitions):
            l12 = transitions[i].wave
            tau = AbsorptionProfile._getOpticalDepth(l12, l12, log10N, b, transitions[i].f_osc)
            maximafluxes[i] = 1-np.exp(-tau)

        maxheight = np.max(maximafluxes)

        return maxheight, maximafluxes/maxheight
    
    # unit can be A or kms, km/s will multiply eq by c/lambda_12
    def getEquivalentWidth(ion, log10N, b, unit='A'):
        ews = []

        for transition in IonTransitions[ion]:
            l12 = transition.wave
            f12 = transition.f_osc

            def oneminusflux(l):
                return 1-np.exp(-AbsorptionProfile._getOpticalDepth(l, l12, log10N, b, f12))

            ew = quad(oneminusflux, l12-20, l12+20, epsabs=1e-8, epsrel=1e-6, limit=5000, points=l12)[0]

            if unit == 'kms':
                ew *= fid.LIGHT_SPEED/l12
            ews.append(ew)

        return np.array(ews)

# Assume two transitions
def fnOneDoubletP1DOsc(k, ion, log10N, b, C=0):
    _, r = AbsorptionProfile.getRelativeFluxStrengths(ion, log10N, b)
    r = np.min(r)
    A = np.max(AbsorptionProfile.getEquivalentWidth(ion, log10N, b, unit='kms'))
    mu = getIonVelocitySeparations(ion)[ion][0]
    beta = mu/fid.LIGHT_SPEED
    kb2 = (k*b)**2
    return A**2 * (C+1+r**2 + 2*r*np.cos(k*mu) + 2*(1-r**2)*kb2*beta) * np.exp(-kb2) / fid.LIGHT_SPEED

# returns C + psb0*(k/k0)^-gamma
def fnP1DSBSmooth(k, psb0, gamma, k0=1e-3):
    return psb0*np.power(k/k0, -gamma)


def fnFittingCIV(k, psb0, gamma, Nciv, bciv, L=1):
    return fnP1DSBSmooth(k, psb0, gamma) + fnOneDoubletP1DOsc(k, 'C IV', Nciv, bciv) * L

def fnFittingCIVSiIV(k, psb0, gamma, Nciv, bciv, Nsiiv, bsiiv, L=1):
    return fnFittingCIV(k, psb0, gamma, Nciv, bciv) + fnOneDoubletP1DOsc(k, 'Si IV', Nsiiv, bsiiv)* L

def fnFittingCIVSiIVMgII(k, psb0, gamma, Nciv, bciv, Nsiiv, bsiiv, Nmgii, bmgii, L=1):
    return fnFittingCIVSiIV(k, psb0, gamma, Nciv, bciv, Nsiiv, bsiiv) + fnOneDoubletP1DOsc(k, 'Mg II', Nmgii, bmgii)* L











