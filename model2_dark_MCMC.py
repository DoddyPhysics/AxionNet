import emcee
import numpy as np
import math
import naxion
from quasi_observable_data import *

debugging = True

if debugging:
	import time
	#np.random.seed(121)

##############################
# 	Set up your model		 #
##############################

if debugging:
	print 'starting'

# You must be careful here selecting the right 
# parameters for the model, and priors, and match them up  
# in the prior and likelihood functions below.
# This is not idiot proof!

# Model selection
run_name='model2_dark_run1'
nax=20
model=2

# Sampler parameters
# ndim must be correct for the model!
# nsteps is the number of steps before each save
ndim, nwalkers, nsteps = 5, 10, 1
# repeat numiter times
numiter=50000 # iterate the sampler
# total steps = nwalkers*nsteps*numiter

# Priors
kminl,kminu=-9.,-3.
kmaxl,kmaxu=-3.,1.
mminl,mminu=-2.,1.
mmaxl,mmaxu=1.,8.
betamin,betamax=0.,1.

# Starting position
startFile=False
startChainFile='Chains/model2_dark_run1.npy'

##################################
# Starting position of walkers
##################################

if startFile:
	startChain=np.load(startChainFile)
	# Take last sample from each walker as new starting position
	pos=startChain[:,-1,:]
else:
	# Uniform starts
	# Match these to the prior if using e.g. log-flat
	pos = [[np.random.uniform(kminl,kminu),np.random.uniform(kmaxl,kmaxu),
		np.random.uniform(mminl,mminu),np.random.uniform(mmaxl,mmaxu),np.random.uniform(betamin,betamax)] for i in range(nwalkers)]

##############################
# Likelihood and prior functions
##############################

def lnprior(theta):
	kmin,kmax,mmin,mmax,beta = theta
		
	# Flat priors on parameters
	if kminl<kmin<kminu and kmaxl<kmax<kmaxu and mminl<mmin<mminu and mmaxl<mmax<mmaxu and betamin<beta<betamax:
		if debugging:
			print 'lnprior = ', 0.0
		return 0.0
	if debugging:
		print 'lnprior = ', 'inf'
	return -np.inf

##########################################
# 		The Likelihood					#
#########################################

def lnlike(theta, H0,sigH, Om,sigOm):
		
	kmin,kmax,mmin,mmax,beta = theta

	if debugging:
		start = time.time()
		print 'in likelihood, params   ',theta
	# Initialise the naxion model
	# Hypervec must be correct for the model number, and match the params in theta
	# There is probably an idiot proof way to do this, but for now you have to think!
	my_calculator = naxion.hubble_calculator(ifsampling=True,fname='configuration_card_dark.ini',mnum=model,hypervec=(nax,kmin,kmax,mmin,mmax,beta))

	###############################################################################	
	# Apply a "prior" to the log10(masses)
	# Done inside the likelihood to assure it is same random seed as for solver
	# If mass cut failed return zero likelihood, lnlik=-np.inf

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)

	for i in range(nax):
		if masses[i]>mcut:
			if debugging:
				print 'masses outside prior,  ', np.log10(masses)
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

	#############################################################################

	# Return the product likelihood for Hubble, Om, acc, zeq
	if debugging:
		print 'in likelihood, obs=   ',	Hout,Omout,add0,zout
		end = time.time()
		print 'elapsed time in  lik =   ',end-start	
		print 'lnlik = ', lnlikH+lnlikOm+lnlikacc+lnlikZ
	return lnlikH+lnlikOm+lnlikacc+lnlikZ

#############################################################################


def lnprob(theta, H0,sigH, Om,sigOm):
	lp = lnprior(theta)
	
	if not np.isfinite(lp):
		# do not call the likelihood if the prior is infinite
		return -np.inf

	return lp + lnlike(theta, H0,sigH, Om,sigOm)


#########################################
# 	      Do the MCMC				   #
########################################


print 'running, iteration =  ',0.,'  of  ',numiter
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm))
if debugging:
	print 'running MCMC'
sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
if debugging:
	print 'making chain'
chain=sampler.chain[:,:,:]
if debugging:
	print 'saving'
np.save('Chains/'+run_name+'.npy',chain)

for i in range(1,numiter):
	print 'running, iteration =  ',i,'  of  ',numiter
	pos=chain[:,-1,:]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm))
	if debugging:
		print 'running MCMC'
	sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
	if debugging:
		print 'making chain'
	temp=sampler.chain[:,:,:]
	chain=np.concatenate((chain,temp),axis=1)
	if debugging:
		print 'saving'
	np.save('Chains/'+run_name+'.npy',chain)
	




