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
- The composition $X_B = \frac{1}{2}(X_B^\alpha+X_B^\beta)$ with $X_B^\alpha = N_B^\alpha / N$ , $X_B^\beta = N_B^\beta / N$ fraction of sites occupied by B atoms (similarly with the A atom $X_A^\alpha$, $X_A^\beta$) is set to be between **$0 \le X_B \le 1$** moreover we have **$X_A = 1- X_B$** 
- The long-range order parameter (LRO) $\eta= \frac{1}{2}(X_B^\alpha-X_B^\beta)$ is set to be between **$-0.5 \le \eta \le 0.5$**, moreover we have that  | $\eta$ | $\le$ min($X_B , X_A $)

Then, the calculations for the enthalpy, entropie and gibbs free energy have been done in the following way :
- The enthalpy of mixing is $\Delta H_{mix}=N \Omega (X_A X_B +\eta^2) $
- With $\Omega $ the interaction parameter, if $\Omega>0$ there is a miscibility gap and if $\Omega<0$ there is ordering. $\Omega=0$ corresponds to the ideal solution.
- The entropy of mixing was calculated with Bragg-Williams-Gorsky configurational entropy : $\Delta S_{mix}=k_B \ln[\frac{(N/2)!}{N_A^\alpha !N_B^\alpha !} \cdot \frac{(N/2)!}{N_A^\beta !N_B^\beta !}]$ which can be writen as $\Delta S_{mix}= \frac{-k_B N}{2}(X_A^\alpha \ln(X_A^\alpha)+X_B^\alpha \ln(X_B^\alpha)+X_A^\beta \ln(X_A^\beta)+ X_B^\beta \ln(X_B^\beta))$
- Finally Gibbs free energy can be written as $\Delta G_{mix} = \Delta H_{mix} - T\Delta S_{mix}$

# Part 2 : Spinodal decomposition : Cahn Hilliard equation

The goal of the second part of the software is to solve Cahn Hilliard equation in order to simulate the spinodal decomposition in a virtual A-B alloy. This simulation is a two-phase field simulation where the field variables are assumed to be continuous across the interfacial regions. 

The summary of the theory is based on the course Microscopic Kinetics and Thermodynamics from Prof. Pasquini at the University of Bologna. Moreover, the code was inspired by the model that has been developed by the Yamanaka research group of Tokyo University of Agriculture and Technology in August 2019 (see https://web.tuat.ac.jp/~yamanaka/pcoms2019/Cahn-Hilliard-2d.html). To have more informations about the model one can see the article _J. W. Cahn and J. E. Hilliard, "Free Energy of a Non-Uniform System in Interfacial Energy" Journal of Chemical Physics, 28 (1958), pp. 258-267_ as reference.

The Cahn Hilliard equation governs the kinetics of concerved order parameters such as $c_B$ which is the composition of B atoms in a region, which is the concentration of B atom in atomic fraction. As in the first part some parameters are left to the user.

The total free energy of the system can be writen as : 
- $G=\int_V (g_{chem}(c) + g_{grad}(\nabla c)) dV$ with $g_{chem}$ and $g_{grad}$ the chemical free energy and the gradient energy densities. The temperature T in [K] can be entered by the user. **(for example T=673 [K])**

The regular solution approximation is used to calculate the chemical free energy density with :
- $g_{chem}=RT [c\ln(c) + (1-c)\ln(1-c)]+ Lc(1-c)]$ were L is the atomic interaction parameter $L=\Omega$ that we will set as constant for a set temperature T. This quantity will be calculated, for example if T=673 K is entered we will have L=13943. 
- $g_{grad}=\frac{a_c}{2}|\nabla c|^2$ with $a_c$ the gradient energy coefficient in [Jm2/mol]. In the programme it is named A and can be set by the user and is not related to any physical values. **(for example A=3.0e-14 [Jm2/mol])**

Before solving Cahn Hilliards equation to get the concentration in function of time, a random initial state is generated.

- The average chemical composition of B atoms $c_0$ is set by the user and must be between 0 and 1. **(for example $c_0=0.5$, which means that there are as many A atoms as B atoms)**

- The initial distribution of c is determines as $c_0$ + uniform random number. C is then a matrix of size $N_x*N_y$. The user can set the size of the two phase field matrix. in order for it to make physical sens we set $N_x = N_y$. **(for example Nx=Ny=20)** 

- Moreover, the chemical free energy density curve and the place of the average chemical composition $c_0$ will be plotted near this initial distribution.

The time evolution of the concerved order parameter c follows the Cahn Hilliard equation : $\frac{\partial c}{\partial t} =\nabla \cdot (M_c \nabla \mu)$. 
- Where $\mu = \frac{\delta G}{\delta c}$ is the diffusion potential of B atoms. Moreover, as the functional dericative of G is given by the Euler-Lagrange equation we have $\frac{\delta G}{\delta c} = \frac{\partial g}{\partial c} - \nabla \cdot \frac{\partial g}{\partial (\nabla c)}$

- $M_c$ is the diffusion mobility of B atoms which is given by : $M_c=[\frac{D_A}{RT}c + \frac{D_B}{RT}(1-c)]$. Where $D_A$ and $D_B$ are the diffusion coefficients of atoms A and B in [m2/s]. They are calculated by taking in account a coefficient for the diffusion $\text{coeff}_A$ and $\text{coeff}_B$ that can be entered by the user. **(for example  $\text{coeff}_A = 1.0\text{e-}04$ and $\text{coeff}_B = 2.0\text{e-}05$)** And also by considering and activation energy $E_A$ and $E_B$ in [J] that can also be entered by the user. **(for example $E_A = E_B = 3.0\text{e}05$)**

In order to implement this equation, a discretization by simple finite difference method has been performed. The 1st-order Euler method was used for time-integration and the 2nd-order central finite difference method was used for the spatial derivatives (see work of the Yamanaka research group).

- The discretized time evolution equation can then be written by : $c_{i,j}^{t+\Delta t}=c_{i,j}^t + M_c \cdot A_{i,j} + B_{i,j}\cdot C_{i,j}$
- The concentration of B atom at time t and at the computational grid point  (i,j) is $c_{i,j}^{t}$

- $M_c = \frac{D_A}{RT}[c_{i,j}^t+\frac{D_B}{D_A}(1-c_{i,j}^t)]c_{i,j}^t(1-c_{i,j}^t)$ is the discretized diffusion mobility. 

- $A_{i,j}=\frac{\partial^2 \mu}{\partial x^2} + \frac{\partial^2 \mu}{\partial y^2}=\frac{\mu_{i+1,j}^t-2\mu_{i,j}^t+\mu_{i-1,j}^t}{(\Delta x)^2} + \frac{\mu_{i+1,j}^t-2\mu_{i,j}^t+\mu_{i-1,j}^t}{(\Delta y)^2}$

- $B_{i,j}=\frac{\partial M_c}{\partial c}=\frac{D_A}{RT} [(1-\frac{D_B}{D_A})c_{i,j}^t(1-c_{i,j}^t)+(c_{i,j}^t+\frac{D_B}{D_A}(1-c_{i,j}^t))(1-2c_{i,j}^t)]$ 

- $C_{i,j}=\frac{\partial c}{\partial x}\frac{\partial \mu}{\partial x} + \frac{\partial c}{\partial y}\frac{\partial \mu}{\partial y} = \frac{(c_{i+1,j}^t-c_{i-1,j}^t)(\mu_{i+1,j}^t-\mu_{i-1,j}^t)}{4(\Delta x)^2} + \frac{(c_{i,j+1}^t-c_{i,j-1}^t)(\mu_{i,j+1}^t-\mu_{i,j-1}^t)}{4(\Delta y)^2}$

- As we set $\mu_{i,j}^t=\mu_{i,j}^{t,chem}+\mu_{i,j}^{t,chem}$ the diffusion potential of B atom at time t on the computational grid point (i,j).

- Where we have the chemical term of the diffusion potential : $\mu_{i,j}^{t,chem}= RT[\ln(c_{i,j}^t)-\ln(1-c_{i,j}^t)] + L(1-2c_{i,j}^t)$
- And the gradient term of the diffusion potential : $\mu_{i,j}^{t,grad}=-A[\frac{(c_{i+1,j}^t-2c_{i,j}^t+c_{i-1,j}^t)}{(\Delta x)^2} + \frac{(c_{i,j+1}^t-2c_{i,j}^t+c_{i,j-1}^t)}{(\Delta y)^2}]$

The time increment $\Delta t$ in [s] used for each step of the update of the order parameter is calculated from the grid information entered by the user. 
- The user can set the grid spacing in meter dx dans dy such as for a square matrix we have dx=dy. **(for example dx=dy=$2.0\text{e-}09$ m)**
- Then the time interval would be $\Delta t= (dx*dx/D_a)*0.1$ in second. 

Once all those parameters are entered, the user can enter the parameters used for the animation.

- nsteps is the total number of time-steps **(for example nsteps=600)**
- nprint is a divisor of nsteps for which when n in nsteps is a multiplicator of nprint, the concentration at that time t is stored and ploted. **(for example nprint=60)**
- interval in [ms] is the time interval between each fram in the animation 

The last window of the software enables the user to download the results of the simulation in a txt file.
- The user has to enter the path of the directory in which the file will be saved **(for example : C:\user\directory\ )** (it is important to add the ** \ ** after the name of the directory)
- Then the user has to enter the name of the file **(for example : testfile)** (without the .txt !)




