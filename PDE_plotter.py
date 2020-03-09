import numpy as np
import matplotlib.pyplot as plt
import sys
from PDE_solver import CahnHilliard

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <parameters file>")
        quit()
    else:
        infile_parameters = sys.argv[1]

    # Open input file and assinging parameters.
    with open(infile_parameters, "r") as input_file:
        # Read the lines of the input data file.
        line = input_file.readline()
        items = line.split(", ")
        # Lattice size.
        lattice_size = (int(items[0]), int(items[0]))
        mob = float(items[1])   # Mobility.
        a = float(items[2])     # a.
        kappa = float(items[3]) # kappa.
        dx = float(items[4])    # Spatial discretisation.
        dt= float(items[5])     # Temporal discretisation.
        iters = int(items[6])   # Time iterations.
    
    # Create C-H lattice.
    ch_lattice = CahnHilliard(size = lattice_size, mobility = mob, a = a,
                              kappa = kappa, dx = dx, dt = dt)
    # Data storage.
    density_array = np.zeros(lattice_size)
    time_vals = np.zeros(iters)
    density_vals = np.zeros(iters)
    # Simulation begins.
    for step in range(iters):
        print(step)
        # Calculating free energy density.
        for j in range(ch_lattice.size[0]):
            for k in range(ch_lattice.size[1]):
                density_array[j, k] = ch_lattice.calc_free_energy((j, k))
        # Progress time.
        ch_lattice.update_cahn_hilliard()
        # Recording values.
        density_vals[step] = np.sum(density_array)
        print(density_vals[step])
        time_vals[step] = step*dt
    # Plotting.
    ch_lattice.plot_fed(time_vals, density_vals)

    # Writing to a file.
    with open("free_energy.dat", "w+") as f:
        f.writelines(map("{}, {}\n".format, time_vals, density_vals))
main()
