import numpy as np
import matplotlib.pyplot as plt
import sys
from PDE_solver import CahnHilliard
from PDE_solver import Poisson

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
            phi_0 = float(items[8])

        # Create C-H lattice.
        ch_lattice = CahnHilliard(size=lattice_size, mobility=mob, a=a,
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
                density_array = ch_lattice.calc_free_energy()
                # Recording values.
                density_vals.append(np.sum(density_array))
                time_vals.append(step*dt)
            # Evolve time.
            ch_lattice.update_cahn_hilliard()
        # Plotting.
        ch_lattice.plot_fed(time_vals, density_vals)
        # Writing to a file.
        with open("free_energy.dat", "w+") as f:
            f.writelines(map("{}, {}\n".format, time_vals, density_vals))

    if sys.argv[1] == "poisson":
        pass
main()
