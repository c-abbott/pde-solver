from PDE_solver import CahnHilliard
import numpy as np

# Create instance.
ch_lattice = CahnHilliard(size=(100, 100), mobility=0.1, a=0.1, kappa=0.1, dx=1.0, dt=0.1)
# Run animation.
ch_lattice.run_animation(iterations=1000000, it_per_frame=1)
