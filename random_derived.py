"""
Compute derived parameters from random samples from your chain.
We also histogram the masses
Will add histogram of initial phi vals as well

DJEM, April 2017
"""


import numpy as np
import matplotlib.pyplot as plt
import naxion
from quasi_observable_data import *
from numpy import inf


#############################
# Model
###########################
run_name='Mtheory_wide_nax20_DM_run1'
nax=20
model=3

get_derived=True
hist_masses=True

#############################
# set up derived params
##############################

Nsamps=100
nquasi=4
fillval=-1. # pick a val none of the params has
fillvec=np.zeros(nquasi)
fillvec.fill(fillval)
derived=np.zeros((Nsamps,nquasi))

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

for i in range(Nsamps):
	# Get random sample
	j=np.random.randint(nsteps)
	lF,lL,smin,smax,Ntilde=chain[j,:]
	# Initialise calculator
	my_calculator = naxion.hubble_calculator(ifsampling=True,fname='configuration_card_DM.ini',mnum=model,
		hypervec=(nax,10**lF,10**lL,smin,smax,Ntilde))

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)
	massarray[i,:]=masses
	
	if get_derived:
		print 'computing derived params for sample=   ', i 
		if np.logical_not(masses[masses-mcut>0].size):
			# Call solver if all masses pass cut
			my_calculator.solver()
			derived[i,:]=my_calculator.quasiObs()	
		else:
			# otherwise fill derived with fillvec
			derived[i,:]=fillvec
		print derived[i,:]

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
	
	
	
	
	
