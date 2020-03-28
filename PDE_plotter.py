import numpy as np
import matplotlib.pyplot as plt
import sys
from PDE_solver import CahnHilliard
from PDE_solver import Poisson
from PDE_solver import Poisson2D
import time

def main():
    # Cahn-Hilliard free energy plot.
    if sys.argv[1] == "cahn_hilliard":
        infile_parameters = sys.argv[2]

        # Open input file and assinging parameters.
        with open(infile_parameters, "r") as input_file:
            # Read the lines of the input data file.
            line = input_file.readline()
            items = line.split(", ")
            # Lattice size.
            lattice_size = (int(items[0]), int(items[0]))
            mob = float(items[1])       # Mobility.
            a = float(items[2])         # a.
            kappa = float(items[3])     # kappa.
            dx = float(items[4])        # Spatial discretisation.
            dt= float(items[5])         # Temporal discretisation.
            sweeps = int(items[6])      # Time iterations.
            n = int(items[7])           # nth sweep.
            phi_0 = float(items[8])     # Initial phi value.

        # Create C-H lattice.
        CH_Lattice = CahnHilliard(size=lattice_size, mobility=mob, a=a,
                                    kappa=kappa, dx=dx, dt=dt, phi_0=phi_0)
        # Data storage.
        time_vals = []
        density_vals = []

        # Simulation begins.
        for step in range(sweeps):
            print(step)
            # Measure on nth sweep.
            if step % n:
                # Calculate F.E.D.
                density_array = CH_Lattice.calc_free_energy()
                # Recording values.
                density_vals.append(np.sum(density_array))
                time_vals.append(step*dt)
            # Evolve time.
            CH_Lattice.update_phi()
        # Plotting.
        CH_Lattice.plot_fed(time_vals, density_vals)
        # Writing to a file.
        with open("free_energy.dat", "w+") as f:
            f.writelines(map("{}, {}\n".format, time_vals, density_vals))

    # Generate monopole data.
    if sys.argv[1] == "monopole":
        infile_parameters = sys.argv[2]
        # Open input file and assinging parameters.
        with open(infile_parameters, "r") as input_file:
            # Read the lines of the input data file.
            line = input_file.readline()
            items = line.split(", ")
            cubic_size = (int(items[0]), int(items[0]), int(items[0]))
            dx = float(items[1])        # Spatial discretisation.
            dt = float(items[2])        # Temporal discretisation.
            eps = float(items[3])       # Set epsilon to one.
            phi_0 = float(items[4])     # Initial phi value.
            tol = float(items[5])       # Jacobi convegence tolerance.
            alg = str(items[6])         # Algorithm choice.
            omega = float(items[7])     # SOR omega.
        
        # Create Poisson lattice.
        Lattice = Poisson(
            size=cubic_size, dx=dx, dt=dt, eps=eps, phi_0=phi_0, alg=alg, omega=omega)
        # Implement Jacobi algorithm.
        if Lattice.alg == 'jacobi':
            # Create monopole.
            Lattice.create_monopole()
            # Condition for while loop.
            converged = False
            # Timer.
            tic = time.perf_counter()
            # Simulation begins.
            while not converged:
                # Store previous state.
                state = np.array(Lattice.phi)
                # Update state.
                Lattice.phi = Lattice.jacobi_update(Lattice.phi)
                # Check for convegence.
                converged = Lattice.convergence_check(state, Lattice.phi, tol)
            # Collect data.
            Lattice.collect_data(['jacobi_all_data.txt', 'jacobi_pot_data.txt',
                                  'jacobi_vec_data.txt', 'jacobi_dist_data.txt'])
            # Timer.
            toc = time.perf_counter()
            print("Executed script in {} seconds.".format(toc-tic))

        # Implement Gauss-Seidel algorithm.
        elif Lattice.alg == 'gs':
            # Create monopole.
            Lattice.create_monopole()
            # Condition for while loop.
            converged = False
            # Timer.
            tic = time.perf_counter()
            # Simulation begins.
            while not converged:
                # Store previous state.
                state = np.array(Lattice.phi)
                # Update state.
                Lattice.gs_update_3D(Lattice.phi)
                # Check for convergence.
                converged = Lattice.convergence_check(state, Lattice.phi, tol)
            # Collect data
            Lattice.collect_data(
                ['gs_all_data.txt', 'gs_pot_data.txt', 'gs_vec_data.txt', 'gs_dist_data.txt'])
            # Timer.
            toc = time.perf_counter()
            print("Executed script in {} seconds.".format(toc-tic))

    # Determine optimum SOR omega.    
    if sys.argv[1] == "SOR":
        infile_parameters = sys.argv[2]
        # Open input file and assinging parameters.
        with open(infile_parameters, "r") as input_file:
            # Read the lines of the input data file.
            line = input_file.readline()
            items = line.split(", ")
            lattice_size = (int(items[0]), int(items[0]))
            dx = float(items[1])        # Spatial discretisation.
            dt = float(items[2])        # Temporal discretisation.
            eps = float(items[3])       # Set epsilon to one.
            phi_0 = float(items[4])     # Initial phi value.
            tol = float(items[5])       # Jacobi convegence tolerance.
            alg = str(items[6])         # Algorithm choice.
            omega = float(items[7])     # SOR omega.

        omegas = np.arange(1.0, 2.0, 0.01)
        # Data storage.
        sweeps_list = []
        # Timer.
        tic = time.perf_counter()
        # Simulation begins.
        for omega_val in omegas:
            print(omega_val)
            # Make 2D lattice instance.
            Lattice2D = Poisson2D(
                size=lattice_size, dx=dx, dt=dt, eps=eps, phi_0=phi_0, alg=alg, omega=omega_val)
            # Create monopole.
            Lattice2D.create_monopole_2D()
            # Simulation tracking.
            sweeps = 0
            converged = False
            # Simulation begins
            while not converged:
                # Store previous state.
                state = np.array(Lattice2D.phi)
                # Update state.
                Lattice2D.gs_update_2D(Lattice2D.phi)
                # Update sweeps.
                sweeps += 1
                # Check for convergence.
                converged = Lattice2D.convergence_check2D(state, Lattice2D.phi, tol)
            # Store convergence sweeps.
            sweeps_list.append(sweeps)
        # Write file for plotting.
        Lattice2D.write_omega_file(sweeps_list, omegas, 'SOR_omegas.txt')
        # Timer.
        toc = time.perf_counter()
        print("Executed script in {} seconds.".format(toc-tic))
main()
