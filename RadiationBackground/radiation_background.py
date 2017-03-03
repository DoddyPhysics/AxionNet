import numpy as np
import math
from gstar import *

# Do some simple computations to check on the background evolution

# Constants and conversion factors

Pi=math.pi
Mpl=2.435e27 # Mpl in eV, 1/\sqrt{8 \pi G}
MH=2.13e-33 # MH in eV, 100 km s^-1 Mpc^-1
eVtoMsol=1.78e-36/1.99e30 # eV in kg / Msol in kg
eVtominv=1./1.97327e-7
mtoMpc=3.24078e-23 # 1 metre in Mpc



# cosmological params, from Planck 2015
h=0.68
Om=0.32
zeq=3402

TCMB=2.725/1.16e4 # TCMB in eV, from COBE

rhor0=Pi**2./30.*gR(TCMB)*TCMB**4.
omrh2=rhor0/(3.*MH**2.*Mpl**2.)

ai=1.e-8
Ti=TCMB/ai
#print Ti
#print gR(Ti),gR(TCMB) # check there is a no evolution of gstar at our assumed ai
Hi=np.sqrt(omrh2/ai**4.) # ignore gstar evolution, as we do in the code
mmax=3.*Hi # this is the maximum consistent m
ti=ai**2./2.*(omrh2)**-0.5
#print ti
#print mmax*MH



