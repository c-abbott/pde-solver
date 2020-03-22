import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

class CahnHilliard(object):
    """
        Cahn-Hilliard PDE class.
    """
    def __init__(self, size, mobility, a, kappa, dx, dt, phi_0):
        # Simulation parameters.
        self.size = size
        self.mob = mobility
        self.a = a
        self.kappa = kappa
        self.dx = dx
        self.dt = dt
        # Simulation fields.
        self.phi = np.empty(self.size)
        self.mu = np.empty(self.size)
        # Initialise phi field.
        self.construct_phi(phi_0)

    def construct_phi(self, phi_0):
        """
            Initial construction of scalar phi fields.
        """
        #self.phi = np.random.rand(
        #    self.size[0], self.size[1]) * np.random.choice(a=[-1, 1], size=self.size) + phi_0

        # This initialisation is better for noise.
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.phi[i, j] = np.random.randint(-10, 11)/100.0 + phi_0

    
    def laplacian_conv(self, field):
        """
            2D Laplcian using convolution method.
        """
        kernel = [[0.0, 1.0, 0.0],
                  [1.0, -4.0, 1.0],
                  [0.0, 1.0, 0.0]]
        return (signal.convolve2d(field, kernel, boundary='wrap', mode='same'))
        
    def mu_convolve(self):
        """
            Calculating mu field.
        """
        chem_pot = (
            - self.a * self.phi
            + self.a * self.phi**3
            - self.kappa * self.laplacian_conv(self.phi)
        )
        return (chem_pot)
    
    def phi_convolve(self):
        """
            Updating phi field for next time
            step.
        """
        self.phi = self.phi + \
            (self.mob * self.dt / (self.dx**2)) * \
            self.laplacian_conv(self.mu_convolve())

    def update_cahn_hilliard(self):
        """
            Update CH lattice.
        """
        self.mu_convolve()
        self.phi_convolve()

    def calc_free_energy(self):
        """
            Calculates the free energy density
            for the whole lattice.
        """
        # Gradient Calculation.
        grad = np.gradient(self.phi)
        grad_x = grad[0]
        grad_y = grad[1]

        # Free Energy Density.
        fed = (
            - (self.a / 2.0) * self.phi**2 \
            + (self.a / 4.0) * self.phi**4 \
            + (self.kappa / 2.0) * (grad_x**2 + grad_y**2)
        )
        return (fed)

    def pbc(self, indices):
        """
            Applies periodic boundary conditions (pbc) to a
            2D lattice.
        """
        return(indices[0] % self.size[0], indices[1] % self.size[1])
    
    def run_dynamics(self):
        """
            Run 100 sweeps of CH lattice.
        """
        for run in range(100):
            self.phi_convolve()

    def animate(self, *args):
        """
            Animate wrapper function for FuncAnimate
            object.
        """
        #for i in range(self.it_per_frame):
        #    self.update_cahn_hilliard()
        self.run_dynamics()
        self.image.set_array(self.phi)
        return self.image,
    
    def run_animation(self, iterations, it_per_frame):
        """
            Used in partnership with the tester file
            to run the simulation.
        """
        self.it_per_frame = it_per_frame
        self.figure = plt.figure()
        self.image = plt.imshow(self.phi, cmap='seismic', animated=True, interpolation='gaussian')
        self.animation = animation.FuncAnimation(
            self.figure, self.animate, repeat=False, frames=iterations, interval=20, blit=True)
        plt.clim(-1.1, 1.1)
        plt.colorbar()
        plt.show()
    
    def plot_fed(self, x_data, y_data):
        """
            Free energy plotter.
        """
        plt.title('Free Energy vs. Time')
        plt.ylabel('Free Energy [f]')
        plt.xlabel('Time [sweeps]')
        plt.plot(x_data, y_data, color='tab:orange')
        plt.savefig('plots/fed_plot.png')
        plt.show()
    
class Poisson(object):
    """
        A class to be utitilsed to solve
        Poisson's equation.
    """
    def __init__(self, size, dx, dt, eps, phi_0):
        # Simulation parameters.
        self.size = size
        self.dx = dx
        self.dt = dt
        self.eps = eps
        self.build_fields(phi_0)
    
    def build_fields(self, phi_0):
        """
            Initial construction of scalar fields.
        """
        self.phi = np.zeros(self.size, dtype=float)
        self.rho = np.zeros(self.size, dtype=float)
        # Enforce Dirchlect BC on phi.
        self.set_boundary(self.phi, phi_0)
    
    def create_monopole(self):
        """
            Create monopole at centre of n-dim
            cubic lattice.
        """
        self.rho[self.rho.shape[0]//2,
                 self.rho.shape[1]//2, self.rho.shape[2]//2] = 1.0

    def set_boundary(self, array, phi_0):
        """
            Method used to enforce Dirchlect BC
            on cubic lattice.
        """
        # Find Boundaries.
        mask = np.ones(array.shape, dtype=bool)
        mask[array.ndim * (slice(1, -1),)] = False
        # Initialise phi with noise.
        for i in range(self.size[0]):
            for  j in range(self.size[1]):
                for k in range(self.size[2]):
                    if mask[i, j, k] == False:
                        self.phi[i, j, k] = np.random.randint(-10, 11)/100.0 + phi_0
                    # Dirchlect BC.
                    else:
                        self.phi[i, j, k] = 0

    def jacobi_update_phi(self, field):
        """
            Convolutional method to update 
            a scalar field.
        """
        # 3D kernel.
        kernel = [[[0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0]],

                  [[0.0, 1.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0]],

                  [[0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0]]]
        # Return jacobi update.
        return ((signal.fftconvolve(field, kernel, mode='same') / 6.0) + self.rho)

    def get_elec_field(self):
        """
            Method returning the x, y, and z components
            of the electric field seperately.
        """
        E = np.gradient(self.phi)
        return -1*np.array(E)[0], -1*np.array(E)[1], -1*np.array(E)[2]
    
    def convergence_check(self, val1, val2, tol):
        """
            Jacobi algorithm convegence check.
        """ 
        diff = val2 - val1
        if np.all(diff <= tol):
            return True
        else:
            return False

    def collect_data(self):
        all_data = []
        pot_data = []
        vec_data = []

        Ex, Ey, Ez = self.get_elec_field()

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    all_data.append(
                        [i, j, k, self.phi[i, j, k], \
                            Ex[i][j][k], Ey[i][j][k], Ez[i][j][k]])
                    if k == (self.size[2] // 2.0):
                        pot_data.append([i, j, self.phi[i,j,k]])
                        vec_data.append([i, j, Ex[i][j][k], Ey[i][j][k]])

        all_data = np.array(all_data)
        pot_data = np.array(pot_data)
        vec_data = np.array(vec_data)

        np.savetxt('all_data.txt', all_data)
        np.savetxt('pot_data.txt', pot_data)
        np.savetxt('vec_data.txt', vec_data)
