# Software-and-computing-for-applied-physics
Project made for the course Software and Computing for applied physics at the University of Bologna (Unibo) in the first year of the physics master degree

# Part 1 : Multicomponent homogeneous systems
This code and theory summary are based on the course Microscopic Kinetics and Thermodynamics from Prof. Pasquini at the University of Bologna. 

The goal of this first part of the software is to offer a visualization of the atomistic model for binary systems in an A-B solid solution. In the quasi-chemical atomistic model. The assumptions of the model are :

- There are two types of atoms, type A and B, with number of atoms $N_A$ and $N_B$ with $N=N_A + N_B$ the total number of atoms (and sites as the vacancies are not considered). We consider N as Avogadro’s number so that all quantities refer to one mole.
- Each bond has a defined energy, the like bonds are A-A and B-B bonds and the unlike bonds are A_B.
- There are two equivalent sites in the lattice $\alpha$ and $\beta$
- Only nearest-neighbours interactions are taken into account
- Only the configurational Bragg-Williams-Gorsky entropy is considered
- Strain energy due to atomic size mismatch and vibrational entropy are neglected

Some parameters are left to be entered by the user in the interface, in order to calculate the interaction parameter $\Omega$ :
- One can enter the number of neirest neigbours Z **(for example 8 for a bcc structure)**
- The fraction of energy difference in eV that corresponds to : $E=E_{AB}-\frac{1}{2}(E_{AA}+E_{AB})$ where $=E_{AB}$ is the defined energy for the A-B bond **(for example 0.02 eV)**
- The interaction parameter is then $\Omega = Z*E[eV]*1,60218 \text{e-}19$ in [J] 
- The temperature in K at which the gibb's free energy will be calculated in the (X_B, eta) space. **(for example 550 K)**

The ranges for the order parameters are set:
- The composition $X_B = \frac{1}{2}(X_B^\alpha+X_B^\beta)$ with $X_B^\alpha$ , $X_B^\beta$ fraction of sites occupied by B atoms (similarly with the A atom $X_A^\alpha$, $X_A^\beta$) is set to be between **$0 \le X_B \le 1$** moreover we have **$X_A = 1- X_B$** 
- The long-range order parameter (LRO) $\eta= \frac{1}{2}(X_B^\alpha-X_B^\beta)$ is set to be between **$-0.5 \le \eta \le 0.5$**, moreover we have that  | $\eta$ | $\le$ min($X_B , X_A $)

Then, the calculations for the enthalpy, entropie and gibbs free energy have been done in the following way :
- The enthalpy of mixing is $\Delta H_{mix}=N \Omega (X_A X_B +\eta^2) $
- With $\Omega $ the interaction parameter, if $\Omega>0$ there is a miscibility gap and if $\Omega<0$ there is ordering. $\Omega=0$ corresponds to the ideal solution.
- The entropy of mixing was calculated with Bragg-Williams-Gorsky configurational entropy : $\Delta S_{mix}=k_B \ln[\frac{(N/2)!}{N_A^\alpha !N_B^\alpha !} \cdot \frac{(N/2)!}{N_A^\beta !N_B^\beta !}]$
- 


