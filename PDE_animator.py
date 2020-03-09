from PDE_solver import CahnHilliard
import numpy as np

ch_lattice = CahnHilliard(size=(100, 100), mobility=0.1, a=0.1, kappa=0.1, dx=1.0, dt=0.1)
ch_lattice.run_animation(iterations=10000, it_per_frame=1)
