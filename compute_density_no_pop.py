import sys
import numpy as np
from tqdm import tqdm

from interface import Matrix, state, lineage, densityTaTb, stateToNum, Mdim

rho_values = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30], float)
# 0, 1, 2, 3 => rho_i = 0, pop_a = 0, 1, 0, 1, pop_b = 0, 0, 1, 1
rho_i = int(sys.argv[1])
pop_a = -1
pop_b = -1

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

full_rec_coal = np.zeros((dim, dim))
with tqdm(total=dim * dim) as pbar:
    for j in range(dim):
        for k in range(dim):
            full_rec_coal[j, k] = densityTaTb(method, Pinit, lambda t: params(rho_m, t),
                                     j/dt, k/dt, 1/dt, pop_a=-1, pop_b=-1)
            pbar.update(1)

full_model = np.sum(full_rec_coal, axis=0)/dt

np.save(f'eq_mig=0.1_pop_a={-1}_pop_b={-1}_rho_m={rho_m:.3f}_{dim}_{dt}', full_model)
