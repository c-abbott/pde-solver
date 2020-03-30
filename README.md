# pde-solver
PDE_solver.py contains 4 classes:
* CahnHilliard
* Poisson
* Poisson2D
* Maxwell
Each of which contains methods to solve the corresponding equations.

Cahn Hilliard
The Cahn Hilliard equation was solved using the Jacobi algorithm.
1.) To run the Cahn-Hilliard animation, type the following into your
    command line:
    "python PDE_animator.py phi_0"
    - phi_0 is an order parameter indicating the presence of either oil or water.
    - phi_0 = 0.0 corresponds to equal oil and water.
    - phi_0 = ±0.5 correspsonds to a simualtion of drops of (+)water/(-)oil    oil/water base.

2.) The free energy density was then plotted versus time, a figure of which      can be found under 'figures/fed_plot.png'

Poisson
The Gauss-Seidel algorithm paried with successive over relaxation was used to solve the Poisson equation for a monopole placed in the centre of (100,100,100) cubic lattice. A tolerance of 0.001 was used as a convergence test.

The corresponding figures and corresponding datafiles are the following:
1.) 'figures/contour_100_0.001.png' --> 'poisson_data/gs_pot_data.txt'
2.) 'figures/quiver_100_0.001.png' --> 'poisson_data/gs_vec_data.txt'
3.) 'loglog_field.png' --> 'poisson_data/gs_dist_data.txt'
4.) 'loglog_potential.png' --> 'poisson_data/gs_dist_data.txt'

Data in plots 3.) and 4.) were then fitted with a straight line revealing a (gradient = -1.10813, --2.07811) 1/R, and 1/R**2 dependency for the electrostatic potential and electric field repsectively. Details of fitting can be found in the first and third entries to 'fitting.txt'.

Maxwell
The Gauss-Seidel algorithm paried with successive over relaxation was used to solve Maxwell's fourth equation for a centrally placed wire in a (100,100,100) cubic lattice. A tolerance of 0.001 was used as a convergence test.

The corresponding figures and corresponding datafiles are the following:
5.) 'figures/A_contour_100_0.001.png' --> 'maxwell_data/gs_vec_pot_data.txt'
6.) 'figures/quiver_B_100_0.001.png' --> 'maxwell_data/gs_B_field_data.txt'
7.) 'loglog_b_field.png' --> 'maxwell_data/gs_B_dist_data.txt'
8.) 'log_A_dist.png' --> 'poisson_data/gs_B_dist_data.txt'

Data in plot 7.) was then fitted with a straight line revealing a (gradient = -1.06177) 1/R dependency for the magnetic field repsectively. Details of fitting can be found in the final entry to 'fitting.txt'.

Poisson2D
Dimensionality was also reduced by one making a 2D, (100,100) lattice for which the Poisson equation was then solved. We determined the optimal value for omega, the parameter utilised in the SOR algorithm.

A plot of the findings can be found in:
9.) 'figures/SOR_omega_100_0.001.png' --> 'poisson_data/SOR_omegas.txt'