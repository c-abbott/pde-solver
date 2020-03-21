from PDE_solver import CahnHilliard
import numpy as np

# Create instance.
ch_lattice = CahnHilliard(size=(50, 50), mobility=0.1,
                          a=0.1, kappa=0.1, dx=1.0, dt=1.0, phi_0=0.05)
# Run animation.
ch_lattice.run_animation(iterations=100000, it_per_frame=1)