import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CahnHilliard(object):
    def __init__(self, size, mobility, a, kappa, dx, dt):
        self.size = size
        self.mob = mobility
        self.a = a
        self.kappa = kappa
        self.dx = dx
        self.dt = dt
        self.build_matrices()

    def build_matrices(self):
        self.phi = np.random.choice(a=[-1, 1], size=self.size)
        self.mu = np.zeros(self.size)
    
    def discrete_laplacian(self, field, position):
        i, j = position
        laplacian_x = (field[self.pbc(
            (i+1, j))] + field[self.pbc((i-1, j))] - 2 * field[i, j]) / self.dx**2
        laplacian_y = (field[self.pbc(
            (i, j+1))] + field[self.pbc((i, j-1))] - 2 * field[i, j]) / self.dx**2
        return (laplacian_x + laplacian_y)
    
    def discrete_grad(self, field, position):
        i, j = position
        grad_x = (field[self.pbc((i+1, j))] + field[self.pbc((i-1, j))]) / self.dx
        grad_y = (field[self.pbc((i, j+1))] + field[self.pbc((i, j-1))]) / self.dx
        return (grad_x + grad_y)

    def calc_mu(self, position):
        chem_pot = (
                    - self.a * self.phi[position]
                    + self.a * self.phi[position]**3
                    - self.kappa * self.discrete_laplacian(self.phi, position)
                    )
        return (chem_pot)
    
    def calc_free_energy(self, position):
        fed = (
            - (self.a/2.0) * self.phi[position]**2 \
            + (self.a/4.0) * self.phi[position]**4 \
            + (self.kappa/2.0) * self.discrete_grad(self.phi, position)**2
            )
        return(fed)

    #def calc_mu(self, position):
    #    i, j = position
    #    summation = (
    #                self.phi[self.pbc((i+1, j))] + \
    #                self.phi[self.pbc((i-1, j))] + \
    #                self.phi[self.pbc((i, j+1))] + \
    #                self.phi[self.pbc((i, j-1))] - 4 * self.phi[i, j]
    #                )
    #    mu = - self.a * self.phi[i, j] + self.a * (self.phi[i, j])**3 \
    #        - (self.kappa / self.dx**2)*summation
    #    return(mu)
    
    def euler_update(self, position):
        i, j = position
        summation = (self.mu[self.pbc((i-1, j))] + self.mu[self.pbc((i+1, j))]
                  + self.mu[self.pbc((i, j-1))] + self.mu[self.pbc((i, j+1))]
                  - 4 * self.mu[i, j]
                  )
        next_phi = self.phi[i, j] + (self.mob * self.dt / self.dx**2) * summation
        return(next_phi)

    def update_phi(self):
        new_state = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                new_state[i, j] = self.euler_update((i, j))
        self.phi = new_state
    
    def update_mu(self):
        new_state = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                new_state[i, j] = self.calc_mu((i, j))
        self.mu = new_state

    def update_cahn_hilliard(self):
        self.update_mu()
        self.update_phi()

    def pbc(self, indices):
        """
            Applies periodic boundary conditions (pbc) to a
            2D lattice.
        """
        return(indices[0] % self.size[0], indices[1] % self.size[1])
    
    def animate(self, *args):
        for i in range(self.it_per_frame):
            self.update_cahn_hilliard()
        self.image.set_array(self.phi)
        return self.image,
    
    def run_animation(self, iterations, it_per_frame):
        """
            Used in partnership with the tester file
            to run the simulation.
        """
        self.it_per_frame = it_per_frame
        self.figure = plt.figure()
        self.image = plt.imshow(self.phi, cmap='hot', animated=True)
        self.animation = animation.FuncAnimation(
            self.figure, self.animate, repeat=False, frames=iterations, interval=50, blit=True)
        plt.colorbar()
        plt.show()
