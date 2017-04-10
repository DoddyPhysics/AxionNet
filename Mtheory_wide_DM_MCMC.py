import emcee
import numpy as np
import math
import naxion
from quasi_observable_data import *

debugging = True

if debugging:
	import time
	print 'starting'
	#np.random.seed(121)

##############################
# 	Set up your model		 #
##############################
	

# You must be careful here selecting the right 
# parameters for the model, and priors, and match them up  
# in the prior and likelihood functions below.
# This is not idiot proof!

# Model selection
run_name='Mtheory_wide_nax20_DM_run1'
nax=20
model=3

# Sampler parameters
# ndim must be correct for the model!
# nsteps is the number of steps before each save
ndim, nwalkers, nsteps = 5, 20, 1
# repeat numiter times
numiter=25000 # iterate the sampler
# total steps = nwalkers*nsteps*numiter

# Priors
lFmin,lFmax=90.,120.
lLmin,lLmax=-10.,0.
sminl,sminu=10.,100.
smaxl,smaxu=20.,150.
Nmin,Nmax=0.5,10.

# Starting position
startFile=True
startChainFile='Chains/Mtheory_nax20_DM_run1.npy'

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
	pos = [[np.random.uniform(lFmin,lFmax),np.random.uniform(lLmin,lLmax),np.random.uniform(sminl,sminu)
		,np.random.uniform(smaxl,smaxu),np.random.uniform(Nmin,Nmax)] for i in range(nwalkers)]

##############################
# Likelihood and prior functions
##############################

def lnprior(theta):
	lF,lL,smin,smax,N = theta
		
	# Flat priors on parameters
	if lFmin<lF<lFmax and lLmin<lL<lLmax and sminl<smin<sminu and smaxl<smax<smaxu and Nmin<N<Nmax:
		#if debugging:
		#	print 'lnprior = ', 0.0
		return 0.0
	#if debugging:
	#	print 'lnprior = ', 'inf'
	return -np.inf

##########################################
# 		The Likelihood					#
#########################################

def lnlike(theta, H0,sigH, Om,sigOm):
		
	lF,lL,smin,smax,N = theta
	if debugging:
		start = time.time()
		print 'in likelihood, params   ',lF,lL,smin,smax,N
	# Initialise the naxion model
	# Hypervec must be correct for the model number, and match the params in theta
	# There is probably an idiot proof way to do this, but for now you have to think!
	my_calculator = naxion.hubble_calculator(ifsampling=True,
		fname='configuration_card_DM.ini',mnum=model,hypervec=(nax,10.**lF,10.**lL,smin,smax,N))

	###############################################################################	
	# Apply a "prior" to the log10(masses)
	# Done inside the likelihood to assure it is same random seed as for solver
	# If mass cut failed return zero likelihood, lnlik=-np.inf

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)
	#if debugging:
	#	print 'phivals/Mpl   ',my_calculator.phiin_array
	#	print 'log10(masses/mmax)  ',masses-mcut

	for i in range(nax):
		if masses[i]>mcut:
			#if debugging:
			#	print 'MASSES OUTSIDE PRIOR'
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
		#print 'elapsed time in  lik =   ',end-start	
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


#print 'running, iteration =  ',0.,'  of  ',numiter
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm))
#if debugging:
#	print 'running MCMC'
sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
#if debugging:
#	print 'making chain'
chain=sampler.chain[:,:,:]
#if debugging:
#	print 'saving'
np.save('Chains/'+run_name+'.npy',chain)

for i in range(1,numiter):
	#print 'running, iteration =  ',i,'  of  ',numiter
	pos=chain[:,-1,:]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm))
	#if debugging:
	#	print 'running MCMC'
	sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
	#if debugging:
	#	print 'making chain'
	temp=sampler.chain[:,:,:]
	chain=np.concatenate((chain,temp),axis=1)
	#if debugging:
	#	print 'saving'
	np.save('Chains/'+run_name+'.npy',chain)
	




