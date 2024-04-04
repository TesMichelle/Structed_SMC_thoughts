import sys
import numpy as np
from tqdm import tqdm

from interface import Matrix, state, lineage, densityTaTb, stateToNum, Mdim

rho_values = np.array([0.001, 0.2, 0.4, 0.6, 0.8, 1.2, 2.6, 2.8, 3.2, 3.4,
                     3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.5, 6., 6.5, 7.,
                     7.5, 8., 8.5, 9., 9.5, 15., 20.], float)
# 0, 1, 2, 3 => rho_i = 0, pop_a = 0, 1, 0, 1, pop_b = 0, 0, 1, 1
rho_i = int(sys.argv[1]) // 4
pop_a = int(sys.argv[1]) % 2
pop_b = (int(sys.argv[1]) % 4) // 2

rho_m = rho_values[rho_i]
dim = 2000
dt = 100

# явно задаю начальное состояние
Pinit = np.zeros(Mdim)
initial_state = state([lineage(1, 1, 0), lineage(1, 1, 0)])
Pinit[stateToNum[initial_state.n]] = 1.0

# рефернсные параметры из файлы sSMC.ipynb
def params(rho, t):
    return np.array([rho, 1.0, 1.0, 0.1, 0.1])

method = 'expm'

full_rec_coal = np.zeros((4, dim, dim))
c = 0
with tqdm(total=dim * dim) as pbar:
    for j in range(dim):
        for k in range(dim):
            full_rec_coal[c, j, k] = densityTaTb(method, Pinit, lambda t: params(rho_m, t),
                                     j/dt, k/dt, 1/dt, pop_a=pop_a, pop_b=pop_b)
            pbar.update(1)
    c += 1

full_model = np.sum(full_rec_coal, axis=0)/dt

np.save(f'eq_mig=0.1_pop_a={pop_a}_pop_b={pop_b}_rho_m={rho_m:.3f}_{dim}_{dt}', full_model)
