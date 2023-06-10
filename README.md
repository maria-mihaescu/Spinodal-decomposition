# Software-and-computing-for-applied-physics
Project made for the course Software and Computing for applied physics at the university of Bologna (Unibo) in the first year of the physics master degree

# Part 1 : Multicomponent homogeneous systems

The goal of this first part of the software is to offer a visualization of the atomistic model for binary systems in an A-B solid solution. In the quasi-chemical atomistic model. The assumptions of the model are :

-There are two types of atoms, type A and B, with number of atoms $N_A$ and $N_B$ with $N=N_A + N_B$ the total number of atoms (and sites as the vacancies are not considered)
-Each bond has a defined energy, the like bonds are A-A and B-B bonds and the unlike bonds are A_B.
- Only nearest-neighbours interactions are taken into account
- Only the configurational Bragg-Williams-Gorsky entropy is considered
- Strain energy due to atomic size mismatch and vibrational entropy are neglected

Some parameters are left to be entered by the user in the interface, in order to calculate the interaction parameter $\omega$ :
- One can enter the number of neirest neigbours Z **(for example 8 for a bcc structure)**
- The fraction of energy difference in eV that corresponds to : $E=E_{AB}-\frac{1}{2}(E_{AA}+E_{AB})$ where $=E_{AB}$ is the defined energy for the A-B bond **(for example 0.02 eV)**
- The temperature in K at which the gibb's free energy will be calculated in the (X_B, eta) space. **(for example 550 K)**

The ranges for the order parameters are set:
-The composition $X_B = \frac{1}{2}(X_B^\alpha+X_B^\beta)$ with $X_B^\alpha$,$X_B^\beta$ fraction of sites occupied by B atoms (similarly with the A atome $X_A^\alpha$,$X_A^\beta$) is set to be between **$0 \le X_B \ge 1$**
-The long-range order parameter (LRO) $\eta= \frac{1}{2}(X_B^\alpha-X_B^\beta)$ is set to be between **$-0.5 \le \eta \ge 0.5$**, moreover we have that $\abs(\eta) \le min(X_B,X_A)$
