import numpy as np
import sys
import time

sys.path.append('/sjc/quadriga-lib/lib')
import quadriga_lib

print(quadriga_lib.components())


print("Start Python")
t0 = time.time()

P = 1000 # Paths
S = 1000 # Snapshots

#coeff = [np.random.random((4, 4, P)) for _ in range(S)]
coeff = [np.random.random((4, 4, P)) + 1j*np.random.random((4, 4, P)) for _ in range(S)]
delay = [np.random.random((4, 4, P)) for _ in range(S)]

t1 = time.time()
print(f"Random Data, t = {t1-t0:.3f}")

x = quadriga_lib.baseband_freq_response(coeff, delay, 20e9, 1024)

t2 = time.time()
print(f"Total time: {t2-t0:.3f}")

