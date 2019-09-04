"""Script to validate the MPS formulation with non ideal mirror."""
import h5py
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import qutip

import fdtd_spontaneous_emission as fdtd
import low_dim_sys
import open_quant_sys_mirror_mps as oqs_mirror_mps


def emitter_decay_sims() -> None:
    """Runs simulations of emitter decay with non ideal MPS."""
    # Some Parameter values for the system being simulated.
    gamma = 1.0 # Decay rate of the emitter.
    delta = 0.0 # Detuning of emitter frequency from reference frequency.
    delay = 4.0 / gamma # Delay between emitter and mirror.
    ph_mirrors = [0, 0.25 * np.pi,
                  0.5 * np.pi, 0.75 * np.pi, np.pi] # Mirror phase.
    ph_delay = np.pi # Phase corresponding to the delay between emitter to
                     # mirror and back at the reference frequency.

    # Numerical parameters.
    dims = 2 # Dimensionality of each waveguide bin.
    dt = 0.05 / gamma # Time-step. Also used for meshing waveguide in MPS.
    num_tsteps = 1000 # Number of time-steps to perform.
    thresh = 1e-2 # Threshold at which to truncate Schmidth decomposition at
                  # each step of MPS.

    # Open a h5 file for storing simulation results.
    data = h5py.File("results/validation_emitter_decay.h5", "w")
    param_grp = data.create_group("parameters")
    param_grp.create_dataset("gamma", data=gamma)
    param_grp.create_dataset("delta", data=delta)
    param_grp.create_dataset("delay", data=delay)
    param_grp.create_dataset("ph_mirrors", data=ph_mirrors)
    param_grp.create_dataset("ph_delay", data=ph_delay)
    param_grp.create_dataset("dims", data=dims)
    param_grp.create_dataset("dt", data=dt)
    param_grp.create_dataset("num_tsteps", data=num_tsteps)
    param_grp.create_dataset("thresh", data=thresh)
    sim_res_grp = data.create_group("simulation_results")

    # Quick simulation with FDTD. Want to check if the mirror phase values give
    # visibly different results in the decay of the emitter.
    print("Running ideal mirror FDTD simulations.")
    results_fdtd = []
    for ph_mirror in ph_mirrors:
        result = fdtd.tls_ideal_mirror(gamma, delay, ph_delay, ph_mirror,
                                       dt, num_tsteps)
        # Leaving out the last element in the simulation for the length of FDTD
        # result to be consistent with the length of MPS simulation.
        results_fdtd.append(result[:-1])
    sim_res_grp.create_dataset("fdtd", data=np.array(results_fdtd))

    # MPS with ideal mirror.
    print("Running ideal mirror MPS simulations.")
    results_ideal = []
    for ph_mirror in ph_mirrors:
        print("On ph_mirror = {}".format(ph_mirror))
        # Define the system.
        sys = low_dim_sys.TwoLevelSystem(gamma, delta)
        mps_ideal = oqs_mirror_mps.WgIdealMirror(qutip.basis(2, 1),
                                                 delay, ph_mirror,
                                                 ph_delay, dims, sys, dt,
                                                 thresh)
        result = mps_ideal.simulate(num_tsteps,
                                    [qutip.create(2) * qutip.destroy(2)], 50)
        results_ideal.append(result[0])
    sim_res_grp.create_dataset("mps_ideal", data=np.array(results_ideal))

    # MPS with non-ideal mirror MPS code but with reflectivity = 1.
    print("Running nonideal mirror MPS simulations.")
    results_nonideal = []
    for ph_mirror in ph_mirrors:
        print("On ph_mirror = {}".format(ph_mirror))
        # Define the system.
        sys = low_dim_sys.TwoLevelSystem(gamma, delta)
        mps_ideal = oqs_mirror_mps.WgPartialMirror(qutip.basis(2, 1),
                                                   delay, 1.0, ph_mirror,
                                                   ph_delay, dims, sys, dt,
                                                   thresh)
        result = mps_ideal.simulate(num_tsteps,
                                    [qutip.create(2) * qutip.destroy(2)], 50)
        results_nonideal.append(result[0])
    sim_res_grp.create_dataset("mps_nonideal", data=np.array(results_nonideal))
    data.close()


def emitter_drive_sims() -> None:
    """Run simulations of emitter near mirror with a coherent drive."""
    # Some Parameter values for the system being simulated.
    gamma = 1.0 # Decay rate of the emitter.
    delta = 0.0 # Detuning of emitter frequency from reference frequency.
    delay = 4.0 / gamma # Delay between emitter and mirror.
    ph_mirrors = [0, 0.25 * np.pi,
                  0.5 * np.pi, 0.75 * np.pi, np.pi] # Mirror phase.
    ph_delay = np.pi # Phase corresponding to the delay between emitter to
                     # mirror and back at the reference frequency.
    pulse_width = 2 / gamma
    pulse_amp = 0.1 * gamma

    # Numerical parameters.
    dims = 2 # Dimensionality of each waveguide bin.
    dt = 0.05 / gamma # Time-step. Also used for meshing waveguide in MPS.
    num_tsteps = 1000 # Number of time-steps to perform.
    thresh = 1e-2 # Threshold at which to truncate Schmidth decomposition at
                  # each step of MPS.

    # Open a h5 file for storing simulation results.
    data = h5py.File("results/validation_emitter_drive_exponential.h5", "w")
    param_grp = data.create_group("parameters")
    param_grp.create_dataset("gamma", data=gamma)
    param_grp.create_dataset("delta", data=delta)
    param_grp.create_dataset("delay", data=delay)
    param_grp.create_dataset("ph_mirrors", data=ph_mirrors)
    param_grp.create_dataset("ph_delay", data=ph_delay)
    param_grp.create_dataset("dims", data=dims)
    param_grp.create_dataset("dt", data=dt)
    param_grp.create_dataset("num_tsteps", data=num_tsteps)
    param_grp.create_dataset("thresh", data=thresh)
    param_grp.create_dataset("pulse_width", data=pulse_width)
    param_grp.create_dataset("pulse_amp", data=pulse_amp)
    sim_res_grp = data.create_group("simulation_results")

    # Define the pulse.
    def square_pulse(t: float) -> float:
        if t < pulse_width:
            return pulse_amp
        else:
            return 0

    def gaussian_pulse(t: float) -> float:
        return pulse_amp * np.exp(-(t - pulse_cen)**2 / pulse_width**2)

    def exponential_pulse(t: float) -> float:
        return pulse_amp * np.exp(-t / pulse_width)

    # Run the ideal MPS simulations.
    print("Running ideal MPS simulations.")
    results_mps_ideal = []
    for ph_mirror in ph_mirrors:
        print("On ph_mirror = {}".format(ph_mirror))
        # Define the low dimensional system.
        sys = low_dim_sys.DrivenTwoLevelSystem(gamma, delta, exponential_pulse)
        mps_ideal = oqs_mirror_mps.WgIdealMirror(qutip.basis(2, 0), delay,
                                                 ph_mirror, ph_delay, dims,
                                                 sys, dt, thresh)
        result = mps_ideal.simulate(num_tsteps,
                                    [qutip.create(2) * qutip.destroy(2)], 50)
        results_mps_ideal.append(result[0])
    sim_res_grp.create_dataset("mps_ideal", data=np.array(results_mps_ideal))

    # Run the nonideal MPS simulations.
    print("Running nonideal MPS simulations.")
    results_mps_nonideal = []
    for ph_mirror in ph_mirrors:
        print("On ph_mirror = {}".format(ph_mirror))
        # Define the low dimensional system.
        sys = low_dim_sys.DrivenTwoLevelSystem(gamma, delta, exponential_pulse)
        mps_nonideal = oqs_mirror_mps.WgPartialMirror(qutip.basis(2, 0), delay,
                                                      1, ph_mirror, ph_delay,
                                                      dims, sys, dt, thresh)
        result = mps_nonideal.simulate(num_tsteps,
                                       [qutip.create(2) * qutip.destroy(2)],
                                       50)
        results_mps_nonideal.append(result[0])
    sim_res_grp.create_dataset("mps_nonideal",
                               data=np.array(results_mps_nonideal))





if __name__ == "__main__":
    emitter_decay_sims()
    emitter_drive_sims()

