import numpy as np
import sys

from interface import comute_cov


m1 = float(sys.argv[1])
m2 = float(sys.argv[2])

dim = 400
dt = 20

coal_rec = np.load(f'm1={m1}_m2={m2}_{dim}_{dt}_b.npy')

TV_simple = compute_TV(coal_rec, simple_smc)
TV_struct = compute_TV(coal_rec, struct_smc)

print(TV_simple.shape)
np.save('TV_simple', TV_simple)
np.save('TV_struct', TV_struct)
