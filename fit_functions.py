import numpy as np
from scipy.special import erfcx


def R_sq(poly, x, y):
    '''Calculate the coefficient of determination (R²) for a polynomial fit.'''
    y_pred = poly(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def gaussian(x, amp1, cen1, sig1):
    '''Calculate the Gaussian function with amplitude amp1, center cen1, and standard deviation sig1.'''
    norm = sig1 * np.sqrt(2 * np.pi)
    return amp1 * np.exp(-((x - cen1) ** 2) / (2 * sig1 ** 2)) / norm


def two_gaussians(x, amp1, cen1, sig1, amp2, cen2, sig2):
    '''Calculate the sum of two Gaussian functions.'''
    g1 = gaussian(x, amp1, cen1, sig1)
    g2 = gaussian(x, amp2, cen2, sig2)
    return g1 + g2


def pseudo_voigt(x, amp, cen, sig, eta):
    '''Calculate the pseudo-Voigt function, which is a linear combination of a Gaussian and a Lorentzian.'''
    norm_gauss = sig * np.sqrt(2 * np.pi)
    norm_lorentz = np.pi * sig
    gaussian_part = amp * np.exp(-((x - cen) ** 2) / (2 * sig ** 2)) / norm_gauss
    lorentzian_part = amp * (sig ** 2 / ((x - cen) ** 2 + sig ** 2)) / norm_lorentz
    return eta * lorentzian_part + (1 - eta) * gaussian_part


def two_pseudo_voigts(x, amp1, cen1, sig1, eta1, amp2, cen2, sig2, eta2):
    '''Calculate the sum of two pseudo-Voigt functions.'''
    pv1 = pseudo_voigt(x, amp1, cen1, sig1, eta1)
    pv2 = pseudo_voigt(x, amp2, cen2, sig2, eta2)
    return pv1 + pv2


def EMG(x, h, mu, sigma, tau):
    '''Calculate the exponentially modified Gaussian (EMG) function to model tailing peaks.'''
    prefactor = h * 0.5 * tau
    exp_arg = 0.5 * tau * (2 * mu + tau * sigma ** 2 - 2 * x)
    erfc_arg = (mu + tau * sigma ** 2 - x) / (np.sqrt(2) * sigma)
    return prefactor * np.exp(exp_arg) * erfcx(erfc_arg)


def EMG_mirrored(x, h, mu, sigma, tau):
    return EMG(-x, h, -mu, sigma, tau)


def EMGsTailFront(x, h1, mu1, sigma1, tau1, h2, mu2, sigma2, tau2):
    tail_emg = EMG(x, h1, mu1, sigma1, tau1)
    front_emg = EMG_mirrored(x, h2, mu2, sigma2, tau2)
    return tail_emg + front_emg


def make_constrained_EMG(tau_coeffs, sigma_coeffs):
    def constrained_EMG(x, h, mu):
        tau = tau_coeffs[1] + h * tau_coeffs[0]
        sigma = sigma_coeffs[1] + h * sigma_coeffs[0]

        if sigma <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        prefactor = h * 0.5 * tau
        exp_arg = 0.5 * tau * (2 * mu + tau * sigma ** 2 - 2 * x)
        erfc_arg = (mu + tau * sigma ** 2 - x) / (np.sqrt(2) * sigma)
        return prefactor * np.exp(exp_arg) * erfcx(erfc_arg)

    return constrained_EMG


def make_constrained_EMG_mirrored(tau_coeffs, sigma_coeffs):
    def constrained_EMG_mirrored(x, h, mu):
        tau = tau_coeffs[1] + h * tau_coeffs[0]
        sigma = sigma_coeffs[1] + h * sigma_coeffs[0]

        if sigma <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        return EMG(-x, h, -mu, sigma, tau)

    return constrained_EMG_mirrored


def make_constrained_EMGsTailFront(
    tail_tau_coeffs,
    tail_sigma_coeffs,
    front_tau_coeffs,
    front_sigma_coeffs,
):
    constrained_EMG = make_constrained_EMG(tail_tau_coeffs, tail_sigma_coeffs)
    constrained_EMG_mirrored = make_constrained_EMG_mirrored(front_tau_coeffs, front_sigma_coeffs)

    def constrained_EMGsTailFront(x, h1, mu1, h2, mu2):
        tail_emg = constrained_EMG(x, h1, mu1)
        front_emg = constrained_EMG_mirrored(x, h2, mu2)
        return tail_emg + front_emg

    return constrained_EMGsTailFront


def make_sigma_constrained_EMG(sigma_coeffs):
    def constrained_EMG(x, h, mu, tau):
        sigma = sigma_coeffs[1] + h * sigma_coeffs[0]

        if sigma <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        prefactor = h * 0.5 * tau
        exp_arg = 0.5 * tau * (2 * mu + tau * sigma ** 2 - 2 * x)
        erfc_arg = (mu + tau * sigma ** 2 - x) / (np.sqrt(2) * sigma)
        return prefactor * np.exp(exp_arg) * erfcx(erfc_arg)

    return constrained_EMG


def make_sigma_constrained_EMG_mirrored(sigma_coeffs):
    def constrained_EMG_mirrored(x, h, mu, tau):
        sigma = sigma_coeffs[1] + h * sigma_coeffs[0]

        if sigma <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        return EMG(-x, h, -mu, sigma, tau)

    return constrained_EMG_mirrored


def make_sigma_constrained_EMGsTailFront(
    tail_sigma_coeffs,
    front_sigma_coeffs,
):
    constrained_EMG = make_sigma_constrained_EMG(tail_sigma_coeffs)
    constrained_EMG_mirrored = make_sigma_constrained_EMG_mirrored(front_sigma_coeffs)

    def constrained_EMGsTailFront(x, h1, mu1, tau1, h2, mu2, tau2):
        tail_emg = constrained_EMG(x, h1, mu1, tau1)
        front_emg = constrained_EMG_mirrored(x, h2, mu2, tau2)
        return tail_emg + front_emg

    return constrained_EMGsTailFront
