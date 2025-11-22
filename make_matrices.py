# make_matrices.py
import numpy as np

A = np.ones((512, 512), dtype=np.float32)
B = np.ones((512, 512), dtype=np.float32) * 2

np.savez("matrix_a.npz", A)
np.savez("matrix_b.npz", B)
print("Saved matrix_a.npz & matrix_b.npz")
