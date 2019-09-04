import numpy as np
from matplotlib import pyplot as plt

import qutip

import open_quant_sys_mirror_mps as oqs_mirror_mps
import low_dim_sys


gamma = 1.0
delta = 0.0
delay = 4.0 / gamma
ph_mirror = 0
ph_delay = np.pi
dt = 0.025
num_tsteps = 400
dims = 2

sys = low_dim_sys.TwoLevelSystem(gamma, delta)
mps = oqs_mirror_mps.WgIdealMirror(qutip.basis(2, 1),
                    delay, ph_mirror, ph_delay, dims, sys, dt)
mps_mirr = oqs_mirror_mps.WgPartialMirror(
        qutip.basis(2, 1), delay, 1.0, ph_mirror, ph_delay, dims, sys, dt)
res_mirr = mps_mirr.simulate(num_tsteps, [qutip.create(2) * qutip.destroy(2)])
res = mps.simulate(num_tsteps, [qutip.create(2) * qutip.destroy(2)])

plt.plot(np.real(res_mirr[0]), label="Non-ideal")
plt.plot(np.real(res[0]), "--k", label="Ideal")
plt.legend(fontsize=20)
plt.show()

