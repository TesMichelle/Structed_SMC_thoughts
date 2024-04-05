import numpy as np
import matplotlib.pyplot as plt
from interface import compute_joint_smc, block_pop, compute_TV


dim = 2000
dt = 100
rho_values = np.array([0.001, 0.2, 0.4, 0.6, 0.8, 1.2, 2.6, 2.8, 3.2, 3.4,
                     3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.5, 6., 6.5, 7.,
                     7.5, 8., 9., 9.5, 15., 20., 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], float)

dens_dx = block_pop(0.001, 2000, 100)

joint_st_smc = compute_joint_smc(dens_dx, rho_values)
joint_st_smc = joint_st_smc[:, :dim, :dim]  \
             + joint_st_smc[:, dim:, dim:] \
             + joint_st_smc[:, dim:, :dim] \
             + joint_st_smc[:, :dim, dim:]

np.save(f'eq_mig=0.1_{dim}_{dt}_smc', joint_st_smc)

dens_dx = np.load(f'eq_mig=0.1_pop_a=-1_pop_b=-1_rho_m=0.001_{dim}_{dt}.npy')
joint_smc = compute_joint_smc(dens_dx, rho_values)
np.save(f'eq_mig=0.1_{dim}_{dt}_simple_smc', joint_smc)

simple_smc = np.load(f'eq_mig=0.1_{dim}_{dt}_simple_smc.npy')
struct_smc = np.load(f'eq_mig=0.1_{dim}_{dt}_smc.npy')
coal_rec = np.load(f'eq_mig=0.1_{dim}_{dt}.npy')

TV_simple = compute_TV(coal_rec, simple_smc)
TV_struct = compute_TV(coal_rec, struct_smc)

print(TV_simple.shape)
np.save('TV_simple', TV_simple)
np.save('TV_struct', TV_struct)
