from PDE_solver import CahnHilliard
import numpy as np
import sys

phi_0 = float(sys.argv[1])
# Create instance.
ch_lattice = CahnHilliard(size=(50, 50), mobility=0.1,
                          a=0.1, kappa=0.1, dx=1.0, dt=1.0, phi_0=phi_0)
# Run animation.
ch_lattice.run_animation(iterations=10000000, it_per_frame=1)