# Competing magnetic fluctuations and orders in a multiorbital model of doped SrCo2As2

[Ana-Marija Nedić](https://amnedic.github.io), Morten H. Christensen, [Peter P. Orth](https://faculty.sites.iastate.edu/porth/)

### Abstract

We revisit the intriguing magnetic behavior of the paradigmatic itinerant frustrated magnet SrCo2As2, which shows strong and competing magnetic fluctuations yet does not develop long-range magnetic order. By calculating the static spin susceptibility $\chi(q)$ within a realistic sixteen orbital Hubbard-Hund model, we determine the leading instability to be ferromagnetic (FM). We then explore the effect of doping and calculate the critical Hubbard interaction strength $U_c$ that is required for the development of magnetic order. We find that $U_c$ decreases under electron doping and with increasing Hund’s coupling $J$, but increases rapidly under hole doping. This suggests that magnetic order could possibly emerge under electron doping but not under hole doping, which agrees with experimental findings. We map out the leading magnetic instability as a function of doping and Hund’s coupling and find several antiferromagnetic phases in addition to FM. We also quantify the degree of itinerant frustration in the model and resolve the contributions of different orbitals to the magnetic susceptibility. Finally, we discuss the dynamic spin susceptibility, $\chi(q, ω)$, at finite frequencies, where we recover the anisotropy of the peaks at $Q_{\pi} = (\pi, 0)$ and $(0, \pi)$ observed by inelastic neutron scattering that is associated with the phenomenon of itinerant magnetic frustration. By comparing results between theory and experiment, we conclude that the essential experimental features of doped SrCo2As2 are well captured by a Hubbard-Hund multiorbital model if one considers a small shift of the chemical potential towards hole doping.

### Description
This repository includes information, code, scripts, and data to generate the figures in the paper.

### Requirements

* numpy
* scipy
* matplotlib
* multiprocessing
* numba

### Data Generation
The main files to perform the algorithm detailed in the paper are given in folder **main_codes** and described below. The following files were designed to be run on a computing cluster, and they may need to be modified to run on other systems. All data necessary to generate the results are given in the folder **Wannier90**. The partial generated data is given in the folder **results**.

* `multiorbital_bare_susceptibility.py` calculates all static and dynamic transverse bare spin susceptibility tensor components. The code consists of two options: the original Wannier90 tight-binding or a symmetrized Hamiltonian.
* `multiorbital_RPA_susceptibility.ipynb` calculates the RPA susceptibility tensor from the static bare susceptibility and generates the phase diagram from the leading energy eigenvalue.

### Figures
All the codes used to create the figures in the paper are found in the **figures_scripts** folder. They are all written in Python (as jupyter notebooks and also as pure python code) and Mathematica. Used libraries include matplotlib, numpy, scipy, csv and time.
