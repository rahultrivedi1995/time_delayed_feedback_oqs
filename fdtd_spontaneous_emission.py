"""Implementing spontaneous emission computation for feedback system."""
import numpy as np


def tls_ideal_mirror(gamma: float,
                     delay: float,
                     ph_delay: float,
                     ph_mirror: float,
                     dt: float,
                     num_tsteps: int) -> np.ndarray:
    """Simulate spontaneous emission from TLS coupling to waveguide with mirror.

    Args:
        gamma: The decay rate of the mirror.
        delay: The delay between the emitter and the mirror and back.
        ph_delay: The phase corresponding to the resonant frequency from the
            emitter to mirror and back.
        ph_mirror: The phase imparted by the mirror.

    Returns:
        The excited state probability as a function of time.
    """
    # The number time-steps corresponding to the delay.
    delay_num_tsteps = int(delay // dt)
    # Total phase due to mirror and delay.
    phase = ph_delay + ph_mirror

    # Initializing the amplitude of the excited state.
    amp_e = np.zeros(num_tsteps + 1, dtype=complex)
    amp_e[0] = 1.0
    for tstep in range(1, num_tsteps + 1):
        # Compute the feedback term. Note that the feedback term is 0 if the
        # `tstep` is smaller than the time-steps corresponding to the delay
        # `delay_num_tsteps`.
        if tstep < delay_num_tsteps:
            fb = 0
        else:
            fb = amp_e[tstep - delay_num_tsteps]

        # Update the excited state.
        amp_e[tstep] = np.exp(-0.5 * gamma * dt) * (
                amp_e[tstep - 1] - 0.5 * dt * gamma * np.exp(1.0j * phase) * fb)

    return np.abs(amp_e)**2


def tls_partial_mirror(gamma: float,
                       delay: float,
                       ph_delay: float,
                       coup_const: complex,
                       dt: float,
                       num_tsteps: int) -> np.ndarray:
    """Simulate spontaneous emission from TLS coupling to waveguide with mirror.

    Args:
        gamma: The decay rate of the mirror.
        delay: The delay between the emitter and the mirror and back.
        ph_delay: The phase corresponding to the resonant frequency from the
            emitter to mirror and back.
        coup_const: The coupling constant corresponding to the mirror.
        dt: The time-step used for discretizing FDTD.
        num_tsteps: The number of time steps to use in FDTD.

    Returns:
        The excited state probability as a function of time.
    """
    # The number time-steps corresponding to the delay.
    delay_num_tsteps = int(delay // dt)

    # Initializing the amplitude of the excited state.
    amp_e = np.zeros(num_tsteps + 1, dtype=complex)
    amp_e[0] = 1.0
    for tstep in range(1, num_tsteps + 1):
        # Compute the feedback term. Note that the feedback term is 0 if the
        # `tstep` is smaller than the time-steps corresponding to the delay
        # `delay_num_tsteps`.
        if tstep < delay_num_tsteps:
            fb = 0
        else:
            prefac = 1.0j * coup_const.conj() / (
                    1 + 0.25 * np.abs(coup_const**2))
            fb = prefac * amp_e[tstep - delay_num_tsteps]

        # Update the excited state.
        amp_e[tstep] = np.exp(-0.5 * gamma * dt) * (
                amp_e[tstep - 1] + 0.5 * dt * gamma * np.exp(
                        1.0j * ph_delay) * fb)

    return np.abs(amp_e)**2


