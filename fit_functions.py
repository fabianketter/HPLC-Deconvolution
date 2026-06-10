import numpy as np
from scipy.special import erfc


def r_squared(model, x, y):
    '''
    Calculate the coefficient of determination (R²) for a model fit.
    Parameters:
    model : callable
        The fitted model.
    x : array-like
        The independent variable data points.
    y : array-like
        The dependent variable data points.
    Returns:
    R² : float
        The coefficient of determination.
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    y_pred = model(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def gaussian(x, h, cen, sig):
    '''
    Calculate a Gaussian function with area under the curve h, center cen, and standard deviation sig.
    Parameters:
    x : array-like
        The independent variable data points.
    h : float
        The area under the Gaussian curve.
    cen : float
        The center of the Gaussian function.
    sig : float
        The standard deviation of the Gaussian function.
    Returns:
    gaussian : np.ndarray
        The Gaussian function values at x.
    '''
    x = np.asarray(x)
    if sig <= 0:
        return np.full_like(x, np.nan, dtype=float)
    norm = sig * np.sqrt(2 * np.pi)
    return h * np.exp(-((x - cen) ** 2) / (2 * sig ** 2)) / norm


def two_gaussians(x, h1, cen1, sig1, h2, cen2, sig2):
    '''
    Calculate the sum of two Gaussian functions.
    Parameters:
    x : array-like
        The independent variable data points.
    h1 : float
        The area under the first Gaussian function.
    cen1 : float
        The center of the first Gaussian function.
    sig1 : float
        The standard deviation of the first Gaussian function.
    h2 : float
        The area under the second Gaussian function.
    cen2 : float
        The center of the second Gaussian function.
    sig2 : float
        The standard deviation of the second Gaussian function.
    Returns:
    two_gaussians : np.ndarray
        The sum of the two Gaussian functions at x.
    '''
    x = np.asarray(x)
    g1 = gaussian(x, h1, cen1, sig1)
    g2 = gaussian(x, h2, cen2, sig2)
    return g1 + g2


def pseudo_voigt(x, h, cen, sig, eta):
    '''
    Calculate the pseudo-Voigt function, which is a linear combination of a Gaussian and a Lorentzian.
    Parameters:
    x : array-like
        The independent variable data points.
    h : float
        The area under the pseudo-Voigt function.
    cen : float
        The center of the pseudo-Voigt function.
    sig : float
        The standard deviation of the pseudo-Voigt function.
    eta : float
        The mixing parameter between the Gaussian and Lorentzian components.
    Returns:
    pseudo_voigt : np.ndarray
        The pseudo-Voigt function values at x.
    '''
    x = np.asarray(x)
    if sig <= 0 or not (0 <= eta <= 1):
        return np.full_like(x, np.nan, dtype=float)
    norm_gauss = sig * np.sqrt(2 * np.pi)
    norm_lorentz = np.pi * sig
    gaussian_part = h * np.exp(-((x - cen) ** 2) / (2 * sig ** 2)) / norm_gauss
    lorentzian_part = h * (sig ** 2 / ((x - cen) ** 2 + sig ** 2)) / norm_lorentz
    return eta * lorentzian_part + (1 - eta) * gaussian_part


def two_pseudo_voigts(x, h1, cen1, sig1, eta1, h2, cen2, sig2, eta2):
    '''
    Calculate the sum of two pseudo-Voigt functions.
    Parameters:
    x : array-like
        The independent variable data points.
    h1 : float
        The area under the first pseudo-Voigt function.
    cen1 : float
        The center of the first pseudo-Voigt function.
    sig1 : float
        The standard deviation of the first pseudo-Voigt function.
    eta1 : float
        The mixing parameter of the first pseudo-Voigt function.
    h2 : float
        The area under the second pseudo-Voigt function.
    cen2 : float
        The center of the second pseudo-Voigt function.
    sig2 : float
        The standard deviation of the second pseudo-Voigt function.
    eta2 : float
        The mixing parameter of the second pseudo-Voigt function.
    Returns:
    two_pseudo_voigts : np.ndarray
        The sum of the two pseudo-Voigt functions at x.
    '''
    x = np.asarray(x)
    pv1 = pseudo_voigt(x, h1, cen1, sig1, eta1)
    pv2 = pseudo_voigt(x, h2, cen2, sig2, eta2)
    return pv1 + pv2


def EMG(x, h, mu, sig, tau):
    '''
    Calculate the exponentially modified Gaussian (EMG) function to model tailing peaks.
    Parameters:
    x : array-like
        The independent variable data points.
    h : float
        The area under the EMG function.
    mu : float
        The center of the EMG function.
    sig : float
        The standard deviation of the EMG function.
    tau : float
        The exponential parameter of the EMG function.
    Returns:
    EMG : np.ndarray
        The EMG function values at x.
    '''
    x = np.asarray(x)
    if sig <= 0 or tau <= 0:
        return np.full_like(x, np.nan, dtype=float)
    prefactor = h * 0.5 * tau
    exp_arg = 0.5 * tau * (2 * mu + tau * sig ** 2 - 2 * x)
    exp_arg = np.clip(exp_arg, -500, 500)
    erfc_arg = (mu + tau * sig ** 2 - x) / (np.sqrt(2) * sig)
    A  = np.exp(exp_arg)
    B = erfc(erfc_arg)
    return prefactor * A * B




def EMG_mirrored(x, h, mu, sig, tau):
    '''
    Calculate the mirrored exponentially modified Gaussian (EMG) function to model fronting peaks.
    Parameters:
    x : array-like
        The independent variable data points.
    h : float
        The area under the EMG function.
    mu : float
        The center of the EMG function.
    sig : float
        The standard deviation of the EMG function.
    tau : float
        The exponential parameter of the EMG function.
    Returns:
    EMG_mirrored : np.ndarray
        The mirrored EMG function values at x.
    '''
    return EMG(-x, h, -mu, sig, tau)


def EMGsTailFront(x, h1, mu1, sig1, tau1, h2, mu2, sig2, tau2):
    '''
    Calculate the sum of an exponentially modified Gaussian (EMG) function for tailing and a mirrored EMG function for fronting.
    Parameters:
    x : array-like
        The independent variable data points.
    h1 : float
        The area under the EMG function.
    mu1 : float
        The center of the EMG function.
    sig1 : float
        The standard deviation of the EMG function.
    tau1 : float
        The exponential parameter of the EMG function.
    h2 : float
        The area under the mirrored EMG function.
    mu2 : float
        The center of the mirrored EMG function.
    sig2 : float
        The standard deviation of the mirrored EMG function.
    tau2 : float
        The exponential parameter of the mirrored EMG function.
    Returns:
    EMGsTailFront : np.ndarray
        The sum of the tailing and fronting EMG functions at x.
    '''
    tail_emg = EMG(x, h1, mu1, sig1, tau1)
    front_emg = EMG_mirrored(x, h2, mu2, sig2, tau2)
    return tail_emg + front_emg


def make_constrained_EMG(tau_coeffs, sig_coeffs):
    '''
    Create a constrained exponentially modified Gaussian (EMG) function where tau and sig are linear functions of the area under the curve h.
    Parameters:
    tau_coeffs : tuple
        The coefficients for the linear function of tau.
    sig_coeffs : tuple
        The coefficients for the linear function of sig.
    Returns:
    constrained_EMG : callable
        A constrained EMG function.
    '''
    def constrained_EMG(x, h, mu):
        tau = tau_coeffs[1] + h * tau_coeffs[0]
        sig = sig_coeffs[1] + h * sig_coeffs[0]

        if sig <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        return EMG(x, h, mu, sig, tau)

    return constrained_EMG


def make_constrained_EMG_mirrored(tau_coeffs, sig_coeffs):
    '''
    Create a constrained mirrored exponentially modified Gaussian (EMG) function where tau and sig are linear functions of the area under the curve h.
    Parameters:
    tau_coeffs : tuple
        The coefficients for the linear function of tau.
    sig_coeffs : tuple
        The coefficients for the linear function of sig.
    Returns:
    constrained_EMG_mirrored : callable
        A constrained mirrored EMG function.
    '''
    def constrained_EMG_mirrored(x, h, mu):
        tau = tau_coeffs[1] + h * tau_coeffs[0]
        sig = sig_coeffs[1] + h * sig_coeffs[0]

        if sig <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        return EMG_mirrored(x, h, mu, sig, tau)

    return constrained_EMG_mirrored


def make_constrained_EMGsTailFront(tail_tau_coeffs, tail_sig_coeffs, front_tau_coeffs, front_sig_coeffs):
    '''
    Create a function that models the sum of a constrained exponentially modified Gaussian (EMG) for tailing and a constrained mirrored EMG for fronting, where tau and sig are linear functions of the area under the curve h.
    Parameters:
    tail_tau_coeffs : tuple
        The coefficients for the linear function of tau for the tailing EMG.
    tail_sig_coeffs : tuple
        The coefficients for the linear function of sig for the tailing EMG.
    front_tau_coeffs : tuple
        The coefficients for the linear function of tau for the fronting EMG.
    front_sig_coeffs : tuple
        The coefficients for the linear function of sig for the fronting EMG.
    Returns:
    constrained_EMGsTailFront : callable
        A function that models the sum of the constrained EMG functions.
    '''
    constrained_EMG = make_constrained_EMG(tail_tau_coeffs, tail_sig_coeffs)
    constrained_EMG_mirrored = make_constrained_EMG_mirrored(front_tau_coeffs, front_sig_coeffs)

    def constrained_EMGsTailFront(x, h1, mu1, h2, mu2):
        tail_emg = constrained_EMG(x, h1, mu1)
        front_emg = constrained_EMG_mirrored(x, h2, mu2)
        return tail_emg + front_emg

    return constrained_EMGsTailFront


def make_sig_constrained_EMG(sig_coeffs):
    '''
    Create a constrained exponentially modified Gaussian (EMG) function where sig is a linear function of the area under the curve h, and tau is a free parameter.
    Parameters:
    sig_coeffs : tuple
        The coefficients for the linear function of sig.
    Returns:
    constrained_EMG : callable
        A constrained EMG function.
    '''
    def constrained_EMG(x, h, mu, tau):
        sig = sig_coeffs[1] + h * sig_coeffs[0]

        if sig <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        return EMG(x, h, mu, sig, tau)

    return constrained_EMG


def make_sig_constrained_EMG_mirrored(sig_coeffs):
    '''
    Create a constrained mirrored exponentially modified Gaussian (EMG) function where sig is a linear function of the area under the curve h, and tau is a free parameter.
    Parameters:
    sig_coeffs : tuple
        The coefficients for the linear function of sig.
    Returns:
    constrained_EMG_mirrored : callable
        A constrained mirrored EMG function.
    '''
    def constrained_EMG_mirrored(x, h, mu, tau):
        sig = sig_coeffs[1] + h * sig_coeffs[0]

        if sig <= 0 or tau <= 0:
            return np.full_like(x, np.nan, dtype=float)

        return EMG_mirrored(x, h, mu, sig, tau)

    return constrained_EMG_mirrored


def make_sig_constrained_EMGsTailFront(tail_sig_coeffs, front_sig_coeffs):
    '''
    Create a function that models the sum of a constrained exponentially modified Gaussian (EMG) for tailing and a constrained mirrored EMG for fronting, where sig is a linear function of the area under the curve h, and tau is a free parameter.
    Parameters:
    tail_sig_coeffs : tuple
        The coefficients for the linear function of sig for the tailing EMG.
    front_sig_coeffs : tuple
        The coefficients for the linear function of sig for the fronting EMG.
    Returns:
    constrained_EMGsTailFront : callable
        A function that models the sum of the constrained EMG functions.
     '''
    constrained_EMG = make_sig_constrained_EMG(tail_sig_coeffs)
    constrained_EMG_mirrored = make_sig_constrained_EMG_mirrored(front_sig_coeffs)

    def constrained_EMGsTailFront(x, h1, mu1, tau1, h2, mu2, tau2):
        tail_emg = constrained_EMG(x, h1, mu1, tau1)
        front_emg = constrained_EMG_mirrored(x, h2, mu2, tau2)
        return tail_emg + front_emg

    return constrained_EMGsTailFront
