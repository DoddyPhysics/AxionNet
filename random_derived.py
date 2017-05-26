"""
Compute derived parameters from random samples from your chain.
We also histogram the masses
Will add histogram of initial phi vals as well

There is now an option to get derived with your chain, so that step can be eliminated.
(Derived params option in latest Mtheory Gaussian MCMC)
This is still useful to histogram the masses from your chain.

DJEM, April 2017
"""


import numpy as np
import matplotlib.pyplot as plt
import naxion
from quasi_observable_data import *
from numpy import inf
import time



#############################
# Model
###########################
run_name='Mtheory_nax20_DM_run1'
nax=20
model=3

get_derived=False
hist_masses=True

##########################
# Do you want to remove masses that fail the cut?
# If you use remove=False in calculator to see full mass spectrum, derived will still work
# but when the mass cut is failed you will get dummy fillvec
##########################
remove=True

#############################
# set up derived params
##############################

Nsamps=1000
nquasi=4
fillval=-1. # pick a val none of the params has
fillvec=np.zeros(nquasi+1)
fillvec.fill(fillval)
derived=np.zeros((Nsamps,nquasi+1))

massarray=np.zeros((Nsamps,nax))

#############################
# Load chain
############################

chain=np.load('Chains/'+run_name+'.npy')
nwalkers, nsteps,ndim = np.shape(chain)
burnin = nsteps/4
# Make sample chain removing burnin 
chain=chain[:,burnin:,:].reshape((-1,ndim))
nsteps=np.shape(chain)[0]
#print np.finfo('d').tiny

##################################
# sample
#############################

def lnlike_lite(theta, H0,sigH, Om,sigOm):
	
	Hout,Omout,add0,zout = theta
	
	# Define Gaussian likelihood on derived quasi observables
	lnlikH = -0.5*((Hout-H0)**2./(sigH**2.) +np.log(2.*np.pi*sigH**2.) )
	lnlikOm = -0.5*((Omout-Om)**2./(sigOm**2.) +np.log(2.*np.pi*sigOm**2.) )
	lnlikZ = -0.5*((zout-zeq)**2./(sigZ**2.) +np.log(2.*np.pi*sigZ**2.) )
	
	# Do a cut for "is the Universe accelerating"
	if add0>0:
		lnlikacc = 0.0
	else:
		lnlikacc = -np.inf

	#############################################################################

	# Return the product likelihood for Hubble, Om, acc, zeq
	
	
	return lnlikH+lnlikOm+lnlikacc+lnlikZ

for i in range(Nsamps):
	# Get random sample
	start = time.time()	
	j=np.random.randint(nsteps)
	lFL3,smin,smax,Ntilde,beta=chain[j,:]
	# Initialise calculator
	my_calculator = naxion.hubble_calculator(ifsampling=True,init_Kdiag=True,remove_masses=remove,fname='configuration_card_DM.ini',mnum=model,
		hypervec=(nax,10**lFL3,sbar,svar,Nbar,Nvar,betaM))

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)
	massarray[i,:]=masses
	
	if get_derived:
		#print 'computing derived params for sample=   ', i 
		if np.logical_not(masses[masses-mcut>0].size):
			# Call solver if all masses pass cut
			my_calculator.solver()
			Hout,Omout,addout,zout=my_calculator.quasiObs()
			lnlik=lnlike_lite(my_calculator.quasiObs(), H0,sigH, Om,sigOm)
			derived[i,:]=Hout,Omout,addout,zout,lnlik
		else:
			# otherwise fill derived with fillvec
			derived[i,:]=fillvec
		print 'parms=   ',derived[i,:]
		end = time.time()
		print 'elapsed time=   ',end-start
if get_derived:
	np.savetxt(run_name+'_derived.dat',derived)		

##############################
# Histogram masses
##############################

if hist_masses:	
	flatmass=massarray.flatten()
	numbins=20
	failcut=flatmass[flatmass-mcut>0] # find fails
	zeromass=flatmass[flatmass==-inf] # find zeros
	print 'number of masses=   ',np.shape(flatmass)[0]
	print 'number of cut masses=  ', np.shape(failcut)[0]
	print 'number of zero masses=  ', np.shape(zeromass)[0]
	flatmass=flatmass[flatmass > -inf] # remove zeros
	flatmass=flatmass[flatmass-mcut<0] # remove fails

	goodmasses=np.shape(flatmass)[0]
	# plot the cut
	x=[mcut,mcut]
	y=[0,goodmasses/numbins]

	plt.hist(flatmass,color='b',bins=numbins)
	plt.plot(x,y,'--k',linewidth=2)	
	plt.show()

	plt.clf()

	y=[0,3.]

	plt.hist(failcut,color='r',bins=numbins)
	plt.plot(x,y,'--k',linewidth=2)	
	plt.show()
	
	
	
	
	
