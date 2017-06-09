import numpy as np
import math
import naxion
from quasi_observable_data import *

run_name='mp_DM_GRID'

##############################
# 	Set up your model		 #
##############################
	

# A grid-based sampler on two variables

# Model selection

nax=20
model=1

# 10^2 samples
nsteps = 100

# number of samples to average over
avsteps = 10

# Fixed params
beta=0.5

mbarmin,mbarmax=6.,8.
fbarmin,fbarmax=-2.,-1.

xlist=np.linspace(mbarmin,mbarmax,nsteps)
ylist=np.linspace(fbarmin,fbarmax,nsteps)

##########################################
# 		The Likelihood					#
#########################################

def lnlike(mbar,fbar):
	
	
	my_calculator = naxion.hubble_calculator(ifsampling=True,
		fname='configuration_card_DM.ini',mnum=model,init_Kdiag=True,remove_masses=True,
		hypervec=(nax,beta,10.**mbar,10.**fbar))

	###############################################################################	
	# Apply a "prior" to the log10(masses)
	# Done inside the likelihood to assure it is same random seed as for solver
	# If mass cut failed return zero likelihood, lnlik=-np.inf
	# Using remove_masses=True this should never happen.
	# Using remove_masses=False you need this for consistency.

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)

	for i in range(nax):

		if masses[i]>mcut:
			return -np.inf

	#############################################################################	
	
	# Solve e.o.m.'s for outputs only if mass cut is passed
	my_calculator.solver()
	# Output quasi-observables
	Hout,Omout,add0,zout=my_calculator.quasiObs()
	
	# Define Gaussian likelihood on derived quasi observables
	lnlikH = -0.5*((Hout-H0)**2./(sigH**2.) +np.log(2.*math.pi*sigH**2.) )
	lnlikOm = -0.5*((Omout-Om)**2./(sigOm**2.) +np.log(2.*math.pi*sigOm**2.) )
	lnlikZ = -0.5*((zout-zeq)**2./(sigZ**2.) +np.log(2.*math.pi*sigZ**2.) )
	
	# Do a cut for "is the Universe accelerating"
	if add0>0:
		lnlikacc = 0.0
	else:
		lnlikacc = -np.inf

	lnlik=lnlikH+lnlikOm+lnlikacc+lnlikZ
	#############################################################################
		
	return Hout,Omout,add0,zout,lnlik

#############################################################################


# do a grid based likelihood on the two varying params

for i in range(nsteps):

	for j in range(nsteps):
		x=xlist[i]
		y=ylist[j]
		
		for k in range(avsteps):
			# Average the quasiobservables over 
			print "i=",i,"   j=",j,"   k=",k
			if k==0:
				H,Om,add,z,lik=lnlike(x,y)
			else:
				Htemp,Omtemp,addtemp,ztemp,liktemp=lnlike(x,y)
				H=H+Htemp
				Om=Om+Omtemp
				add=add+addtemp
				z=z+ztemp
				lik=lik+liktemp
		H=H/avsteps
		Om=Om/avsteps
		z=z/avsteps
		add=add/avsteps
		lik=lik/avsteps
		derivfile=open('GridExamples/'+run_name+'_derived.txt','a')
		derivfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x,y,H,Om,add,z,lik))
		derivfile.close()


	