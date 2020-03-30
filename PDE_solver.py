import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
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
        self.phi = np.zeros(self.size)
        self.mu = np.zeros(self.size)
        # Initialise phi field.
        self.construct_phi(phi_0)

    def construct_phi(self, phi_0):
        """
            Initial construction of scalar phi fields.
        """
        # This initialisation is better for noise.
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.phi[i, j] = np.random.uniform(-0.01, 0.011) + phi_0

    
    def laplacian(self, field):
        """
            2D Laplcian using convolution method.
        """
        kernel = [[0.0, 1.0, 0.0],
                  [1.0, -4.0, 1.0],
                  [0.0, 1.0, 0.0]]
        return (signal.convolve2d(field, kernel, boundary='wrap', mode='same'))
        
    def get_mu(self):
        """
            Calculating mu field.
        """
        chem_pot = (
            - self.a * self.phi
            + self.a * self.phi**3
            - self.kappa * self.laplacian(self.phi)
        )
        return (chem_pot)
    
    def update_phi(self):
        """
            Updating phi field for next time
            step.
        """
        self.phi = self.phi + \
            (self.mob * self.dt / (self.dx**2)) * \
            self.laplacian(self.get_mu())

    def calc_free_energy(self):
        """
            Calculates the free energy density
            for the whole lattice.
        """
        # Gradient Calculation.
        grad = np.gradient(self.phi)
      
        # Free Energy Density.
        fed = (
            - (self.a / 2.0) * self.phi**2 \
            + (self.a / 4.0) * self.phi**4 \
            + (self.kappa / 2.0) * (grad[0]**2 + grad[1]**2)
        )
        return (fed)

    def pbc(self, indices):
        """
            Applies periodic boundary conditions (pbc) to a
            2D lattice.
        """
        return (indices[0] % self.size[0], indices[1] % self.size[1])
    
    def run_dynamics(self):
        """
            Run 100 sweeps of CH lattice.
        """
        for _ in range(100):
            self.update_phi()

    def animate(self, *args):
        """
            Animate wrapper function for FuncAnimate
            object.
        """
        self.run_dynamics()
        self.image.set_array(self.phi)
        return (self.image,)
    
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
    def __init__(self, size, dx, dt, eps, phi_0, alg, omega):
        # Simulation parameters.
        self.size = size
        self.dx = dx
        self.dt = dt
        self.eps = eps
        self.build_fields(phi_0)
        self.alg = alg
        self.omega = omega
    
    def build_fields(self, phi_0):
        """
            Initial construction of scalar fields.
        """
        self.phi = np.zeros(self.size, dtype=float)
        # Enforce Dirchlect BC on phi.
        self.set_boundary(self.phi, phi_0)
        # Domain for monopole.
        self.rho = np.zeros(self.size, dtype=float)

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

    def jacobi_update(self, field):
        """
            Convolutional method to update 
            a scalar field using Jacobi
            algorithm.
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
        return ((signal.fftconvolve(field, kernel, mode='same')  + self.rho)/ 6.0)

    def gs_update_3D(self):
        """
            3D Gauss-Seidel algorithm with successive
            over relaxation for updating a scalar field. 
            Boundaries not touched as Dirchlect BC in place.
        """
        for i in range(1, self.size[0] - 1):
            for j in range(1, self.size[1] - 1):
                for k in range(1, self.size[2] - 1):
                    if (i == 0) or (j == 0) or (k == 0):
                        self.phi[i][j][k] = 0
                    else:
                        self.phi[i][j][k] = 1./6. * (self.phi[i+1][j][k] + self.phi[i-1][j][k] + \
                                                     self.phi[i][j+1][k] + self.phi[i][j-1][k] + \
                                                     self.phi[i][j][k+1] + self.phi[i][j][k-1] + \
                                                     self.rho[i][j][k]) * self.omega + (1 - self.omega) * \
                                                     self.phi[i][j][k]

    def get_elec_field(self):
        """
            Method returning the x, y, and z components
            of the electric field seperately.
        """
        E = np.gradient(self.phi)
        return (-1*np.array(E)[0], -1*np.array(E)[1], -1*np.array(E)[2])
    
    def convergence_check(self, arr1, arr2, tol):
        """
            Jacobi algorithm convegence check.
        """ 
        diff = abs(arr2 - arr1)
        print(np.sum(diff, axis=None))
        if np.sum(diff, axis=None) <= tol:
            return True
        else:
            return False
    
    def calc_radial_dist(self, indices):
        """
            Calculates the distance from the centre
            of a 2D lattice with periodic boundary 
            conditions.
        """
        i, j = indices
        return (math.sqrt((i - self.size[0] / 2)**2 + (j - self.size[1] / 2)**2))

    def collect_data(self, filenames):
        """
            Data collection method for potential contour
            and electric field quiver plots.
        """
        pot_data = []
        vec_data = []
        dist_data = []

        Ex, Ey, Ez = self.get_elec_field()

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    if k == (self.size[2] // 2.0):
                        pot_data.append([i, j, self.phi[i][j][k]])
                        vec_data.append([i, j, Ex[i][j][k], Ey[i][j][k]])
                        dist_data.append([self.calc_radial_dist((i, j)), self.phi[i][j][k], math.sqrt(
                            (Ex[i][j][k])**2+(Ey[i][j][k])**2+(Ez[i][j][k])**2)])

        pot_data = np.array(pot_data)
        vec_data = np.array(vec_data)
        dist_data = np.array(dist_data)

        np.savetxt('poisson_data/' + str(filenames[0]), pot_data)
        np.savetxt('poisson_data/' + str(filenames[1]), vec_data)
        np.savetxt('poisson_data/' + str(filenames[2]), dist_data)

class Poisson2D(object):
    """
        A class to be utitilsed to solve
        Poisson's equation in 2D.
    """
    def __init__(self, size, dx, dt, eps, phi_0, alg, omega):
        # Simulation parameters.
        self.size = size
        self.dx = dx
        self.dt = dt
        self.eps = eps
        self.build_fields2D(phi_0)
        self.alg = alg
        self.omega = omega
    
    def build_fields2D(self, phi_0):
        """
            Initial construction of scalar fields.
        """
        self.phi = np.zeros(self.size, dtype=float)
        # Enforce Dirchlect BC on phi.
        self.set_boundary2D(self.phi, phi_0)
        # Domain for monopole.
        self.rho = np.zeros(self.size, dtype=float)
    
    def create_monopole_2D(self):
        """
            Create monopole at centre of n-dim
            cubic lattice.
        """
        self.rho[self.rho.shape[0]//2, self.rho.shape[1]//2] = 1.0
    
    def set_boundary2D(self, array, phi_0):
        """
            Method used to enforce Dirchlect BC
            on cubic lattice.
        """
        # Find Boundaries.
        mask = np.ones(array.shape, dtype=bool)
        mask[array.ndim * (slice(1, -1),)] = False
        # Initialise phi with noise.
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if mask[i, j] == False:
                    self.phi[i, j] = np.random.randint(-10, 11)/100.0 + phi_0
                # Dirchlect BC.
                else:
                    self.phi[i, j] = 0
    
    def gs_update_2D(self):
        """
            2D Gauss-Seidel algorithm with successive
            over relaxation for updating a scalar field. 
            Boundaries not touched as Dirchlect BC in place.
        """
        for i in range(1, self.size[0] - 1):
            for j in range(1, self.size[1] - 1):
                if (i == 0) or (j == 0):
                    self.phi[i][j] = 0
                else:
                    self.phi[i][j] = 1./4. * (self.phi[i+1][j] + self.phi[i-1][j] +
                                              self.phi[i][j+1] + self.phi[i][j-1] +
                                              self.rho[i][j]) * self.omega + (1 - self.omega) * \
                                              self.phi[i][j]

    def convergence_check2D(self, arr1, arr2, tol):
        """
            Jacobi/GS algorithm convegence check.
        """
        diff = abs(arr2 - arr1)
        if np.sum(diff, axis=None) <= tol:
            return True
        else:
            return False

    def write_omega_file(self, sweeps, omegas, filename):
        """
            File writer.
        """
        with open('poisson_data/' + str(filename), "w+") as f:
            f.writelines(map("{}, {}\n".format,
                             omegas, sweeps))

class Maxwell(object):
    """
        A class to be utitilsed to solve
        one of Maxwell's equations for a static
        Electric field:
        Laplacian(A) = -mu*J
    """
    def __init__(self, size, dx, dt, mu, A_0, alg, omega):
        # Simulation parameters.
        self.size = size
        self.dx = dx
        self.dt = dt
        self.mu = mu
        self.build_fields_mag(A_0)
        self.alg = alg
        self.omega = omega
    
    def build_fields_mag(self, A_0):
        """
            Initial construction of scalar fields.
        """
        self.A = np.zeros(self.size, dtype=float)
        # Enforce Dirchlect BC on phi.
        self.enforce_bc(self.A, A_0)
        # Domain for wire.
        self.J = np.zeros(self.size, dtype=float)

    def enforce_bc(self, array, A_0):
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
                        self.A[i, j, k] = np.random.uniform(-0.01, 0.01) + A_0
                    # Dirchlect BC.
                    else:
                        self.A[i, j, k] = 0
    
    def create_current(self):
        """
            Creates current-carrying wire parallel to 
            the z-axis of a cubic lattice.
        """
        #self.J[:, :, self.J.shape[2]//2] = 1.0
        for k in range(self.size[2]):
            self.J[self.size[0]//2][self.size[1]//2][k] = 1.0
    
    def jacobi_update_mag(self, field):
        """
            Convolutional method to update 
            a scalar field using Jacobi
            algorithm.
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
        return ((signal.fftconvolve(field, kernel, mode='same') + self.J) / 6.0)
    
    def gs_update_mag(self):
        """
            3D Gauss-Seidel algorithm with successive
            over relaxation for updating a scalar field. 
            Boundaries not touched as Dirchlect BC in place.
        """
        for i in range(1, self.size[0] - 1):
            for j in range(1, self.size[1] - 1):
                for k in range(1, self.size[2] - 1):
                    if (i == 0) or (j == 0) or (k == 0):
                        self.A[i][j][k] = 0
                    else:
                        self.A[i][j][k] = 1./6. * (self.A[i+1][j][k] + self.A[i-1][j][k] + \
                                                    self.A[i][j+1][k] + self.A[i][j-1][k] + \
                                                    self.A[i][j][k+1] + self.A[i][j][k-1] + \
                                                    self.J[i][j][k]) * self.omega + (1 - self.omega) * \
                                                    self.A[i][j][k]

    def get_mag_field(self):
        """
            Method returning the x, y, and z components
            of the electric field seperately.
        """
        # Store values.
        grad = np.gradient(self.A)
        # Assign values.
        Bx = grad[1] - grad[2]
        By = grad[2] - grad[0]
        Bz = grad[0] - grad[1]
        # Return values.
        return (Bx, By, Bz)

    def convergence_check(self, arr1, arr2, tol):
        """
            Jacobi algorithm convegence check.
        """
        diff = abs(arr2 - arr1)
        print (np.sum(diff, axis=None))
        if np.sum(diff, axis=None) <= tol:
            return True
        else:
            return False
    
    def calc_radial_dist(self, indices):
        """
            Calculates the distance from the centre
            of a 2D lattice with periodic boundary 
            conditions.
        """
        i, j = indices
        return (math.sqrt((i - self.size[0] / 2)**2 + (j - self.size[1] / 2)**2))
    
    def collect_data(self, filenames):
        """
            Data collection method for potential contour
            and electric field quiver plots.
        """
        pot_data = []
        vec_data = []
        dist_data = []

        Bx, By, Bz = self.get_mag_field()

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    if k == (self.size[2] // 2):
                        pot_data.append([i, j, self.A[i][j][k]])
                        vec_data.append([i, j, Bx[i][j][k], By[i][j][k]])
                        dist_data.append([self.calc_radial_dist((i, j)), self.A[i][j][k], math.sqrt(
                            (Bx[i][j][k])**2+(By[i][j][k])**2+(Bz[i][j][k])**2)])

        pot_data = np.array(pot_data)
        vec_data = np.array(vec_data)
        dist_data = np.array(dist_data)

        np.savetxt('maxwell_data/' + str(filenames[0]), pot_data)
        np.savetxt('maxwell_data/' + str(filenames[1]), vec_data)
        np.savetxt('maxwell_data/' + str(filenames[2]), dist_data)
