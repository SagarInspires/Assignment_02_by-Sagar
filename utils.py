# utils.py
import numpy as np

def angular_freq(f):
    return 2.0 * np.pi * f

def complex_gamma(Rp, Lp, Gp, Cp, freq):
    """
    Complex propagation constant gamma = sqrt((R' + jωL')(G' + jωC'))
    """
    w = angular_freq(freq)
    Z = Rp + 1j * w * Lp
    Y = Gp + 1j * w * Cp
    gamma = np.sqrt(Z * Y)
    return gamma

def characteristic_impedance(Rp, Lp, Gp, Cp, freq):
    """
    Z0 = sqrt((R' + jωL') / (G' + jωC'))
    """
    w = angular_freq(freq)
    Z = Rp + 1j * w * Lp
    Y = Gp + 1j * w * Cp
    Z0 = np.sqrt(Z / Y)
    return Z0

def input_impedance(Z0, ZL, gamma, length):
    """
    Zin = Z0 * (ZL + Z0 tanh(gamma l)) / (Z0 + ZL tanh(gamma l))
    """
    gl = gamma * length
    # np.tanh works on complex
    return Z0 * (ZL + Z0 * np.tanh(gl)) / (Z0 + ZL * np.tanh(gl))

def reflection_coefficient(Z0, ZL):
    """
    Reflection coefficient seen at the load.
    """
    return (ZL - Z0) / (ZL + Z0)

def voltage_current_along_line(Z0, ZL, gamma, length, N=400, Vp=1.0):
    """
    Returns arrays z, V(z), I(z), and Gamma_L.
    Uses forward wave amplitude V+ = Vp, and V- = Gamma_L * V+ * exp(-2*gamma*length)
    """
    Gamma_L = reflection_coefficient(Z0, ZL)
    Vp_val = Vp
    Vm_val = Gamma_L * Vp_val * np.exp(-2 * gamma * length)
    zs = np.linspace(0.0, length, N)
    Vz = Vp_val * np.exp(-gamma * zs) + Vm_val * np.exp(gamma * zs)
    Iz = (Vp_val / Z0) * np.exp(-gamma * zs) - (Vm_val / Z0) * np.exp(gamma * zs)
    return zs, Vz, Iz, Gamma_L

def standing_wave(Rp, Lp, Gp, Cp, freq, ZL_real, ZL_imag, points=400):
    """
    Standing wave patterns (Voltage) for matched, short-circuit, open-circuit,
    or general load cases.
    """
    # Load impedance
    if ZL_real == float("inf"):  # open circuit
        ZL = np.inf
    elif ZL_real == 0 and ZL_imag == 0:  # short circuit
        ZL = 0
    else:
        ZL = complex(ZL_real, ZL_imag)

    # Angular frequency
    omega = 2 * np.pi * freq

    # Propagation constant
    gamma = np.sqrt((Rp + 1j*omega*Lp) * (Gp + 1j*omega*Cp))
    beta = np.imag(gamma)

    # Wavelength
    wavelength = 2 * np.pi / beta if beta != 0 else 1

    # Characteristic impedance
    Z0 = np.sqrt((Rp + 1j*omega*Lp) / (Gp + 1j*omega*Cp))

    # Reflection coefficient
    if ZL == np.inf:   # open circuit
        Gamma = 1
    elif ZL == 0:      # short circuit
        Gamma = -1
    else:
        Gamma = (ZL - Z0) / (ZL + Z0)

    # Assume incident wave amplitude = 1
    V_plus = 1
    z = np.linspace(0, wavelength, points)  # one λ

    # Voltage pattern
    Vz = V_plus * (np.exp(-1j*beta*z) + Gamma * np.exp(1j*beta*z))

    # Normalize axis to λ
    z_norm = z / wavelength

    return z_norm, np.abs(Vz), Gamma
