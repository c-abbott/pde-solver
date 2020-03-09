import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CahnHilliard(object):
    def __init__(self, size, mobility, a, kappa, dx, dt):
        self.size = size
        self.mob = mobility
        self.a = a
        self.kap = kappa
        self.dx = dx
        self.dt = dt
        self.build_matrices()

    def build_matrices(self):
        self.phi = np.random.choice(a=[0, 1], size=self.size)
        self.mu = np.zeros(self.size)
    
    def disc_laplacian(self, field, position):
        i, j = position
        lap_x = (field[self.pbc((i+1, j))] + field[self.pbc((i-1, j))]
                - 2*field[(i, j)]) / self.dx**2
        lap_y = (field[self.pbc((i, j+1))] + field[self.pbc((i, j-1))]
                 - 2*field[(i, j)]) / self.dx**2
        return(lap_x + lap_y)

    def pbc(self, indices):
        """
            Applies periodic boundary conditions (pbc) to a
            2D lattice.
        """
        return(indices[0] % self.size[0], indices[1] % self.size[1])
