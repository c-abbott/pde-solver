# pde-solver
PDE_solver.py contains 4 classes:
* CahnHilliard
* Poisson
* Poisson2D
* Maxwell
Each of which contains methods to solve the corresponding equations.

Cahn Hilliard
The Cahn Hilliard equation was solved using the Jacobi algorithm.
* To run the Cahn-Hilliard animation, type the following into your
  command line:
  "python PDE_animator.py phi_0"

* phi_0 is an order parameter indicating the presence of either oil or water.
* phi_0 = 0.0 corresponds to equal oil and water.
* phi_0 = ±0.5 correspsonds to a simualtion of drops of (+)water/(-)oil    oil/water base.

* The free energy density was then plotted versus time for phi_0 = 0.0 and phi_0 = 0.5 respectively. The corresponding plots and files are:
* 'figures/fed_0.png' --> 'ch_data/free_energy_0.dat'
* 'figures/fed_0.5.png' --> 'ch_data/free_energy_0.5.dat'
It was found that the simulation took approx. 1e6 update sweeps of the lattice to equilibrate for both phi_0 = 0.0 and phi_0 = 0.5 respectively.

Poisson
The Gauss-Seidel algorithm paried with successive over relaxation was used to solve the Poisson equation for a monopole placed in the centre of (100,100,100) cubic lattice. A tolerance of 0.001 was used as a convergence test.

The corresponding figures and corresponding datafiles are the following:
* 'figures/contour_100_0.001.png' --> 'poisson_data/gs_pot_data.txt'
* 'figures/quiver_100_0.001.png' --> 'poisson_data/gs_vec_data.txt'
* 'loglog_field.png' --> 'poisson_data/gs_dist_data.txt'
* 'loglog_potential.png' --> 'poisson_data/gs_dist_data.txt'

Data in plots 'loglog_field.png' and 'loglog_potential.png' were then fitted with a straight line revealing a 1/R, and 1/R**2 (gradient = -1.10813, -2.07811) dependency for the electrostatic potential and electric field repsectively. Details of fitting can be found in the first and third entries to 'fitting.txt'.

Maxwell
The Gauss-Seidel algorithm paried with successive over relaxation was used to solve Maxwell's fourth equation for a centrally placed wire in a (100,100,100) cubic lattice. A tolerance of 0.001 was used as a convergence test.

The corresponding figures and corresponding datafiles are the following:
* 'figures/A_contour_100_0.001.png' --> 'maxwell_data/gs_vec_pot_data.txt'
* 'figures/quiver_B_100_0.001.png' --> 'maxwell_data/gs_B_field_data.txt'
* 'loglog_b_field.png' --> 'maxwell_data/gs_B_dist_data.txt'
* 'log_A_dist.png' --> 'poisson_data/gs_B_dist_data.txt'

Data in plot 'loglog_b_field.png' was then fitted with a straight line revealing a 1/R dependency (gradient = -1.06177) for the magnetic field repsectively. Details of fitting can be found in the final entry to 'fitting.txt'.

Poisson2D
Dimensionality was also reduced by one making a 2D, (100,100) lattice for which the Poisson equation was then solved using the GS+SOR algorithm. Our aim was to determine the optimal value for omega, the parameter utilised in the SOR algorithm. A tolerance of 0.001 was used to test for convergence.

A plot of the findings can be found in:
* 'figures/SOR_omega_100_0.001.png' --> 'poisson_data/SOR_omegas.txt'

An optimal omega values was discovered to be ~ 1.94.
