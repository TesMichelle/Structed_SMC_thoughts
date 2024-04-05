import numpy as np
import sys
from tqdm import tqdm

dim = 2000
dt = 100

rho_values = np.array([0.001, 0.2, 0.4, 0.6, 0.8, 1.2, 2.6, 2.8, 3.2, 3.4,
                     3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.5, 6., 6.5, 7.,
                     7.5, 8., 9., 9.5, 15., 20., 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], float)


dens = np.zeros((rho_values.shape[0], dim, dim))
for i, rho_m in tqdm(enumerate(rho_values)):
    for pop_a in [0, 1]:
        for pop_b in [0, 1]:
            temp_dens = np.load(f'eq_mig=0.1_pop_a={pop_a}_pop_b={pop_b}_rho_m={rho_m:.3f}_{dim}_{dt}.npy')
            dens[i] += temp_dens
np.save(f'eq_mig=0.1_{dim}_{dt}', dens)
