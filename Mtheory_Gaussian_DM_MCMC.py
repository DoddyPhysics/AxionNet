import emcee
import numpy as np
import math
import naxion
from quasi_observable_data import *

debugging = True
derived=True # quick hack for derived params
run_name='Mtheory_Gaussian_nax20_DM_run1'



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

nax=20
model=3

# Sampler parameters
# ndim must be correct for the model!
# nsteps is the number of steps before each save
ndim, nwalkers, nsteps = 6, 20, 1
# repeat numiter times
numiter=25000 # iterate the sampler
# total steps = nwalkers*nsteps*numiter

# Priors
lFL3min,lFL3max=100.,115.
sbarl,sbaru=15.,25.
svarl,svaru=0.1,5.
Nbarl,Nbaru=0.5,1.5
Nvarl,Nvaru=0.01,0.1
betamin,betamax=0.,1.


# Starting position
startFile=False
startChainFile='Chains/Mtheory_Gaussian_nax20_DM_run1.npy'

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
	pos = [[np.random.uniform(lFL3min,lFL3max),np.random.uniform(sbarl,sbaru)
		,np.random.uniform(svarl,svaru),np.random.uniform(Nbarl,Nbaru),np.random.uniform(Nvarl,Nvaru),
		np.random.uniform(betamin,betamax)] for i in range(nwalkers)]

##############################
# Likelihood and prior functions
##############################

def lnprior(theta):
	lFL3,sbar,svar,Nbar,Nvar,beta = theta
		
	# Flat priors on parameters
	if lFL3min<lFL3<lFL3max and sbarl<sbar<sbaru and svarl<svar<svaru and Nbarl<Nbar<Nbaru and Nvarl<Nvar<Nvaru and betamin<beta<betamax:
		if debugging:
			print 'lnprior = ', 0.0
		return 0.0
	if debugging:
		print 'lnprior = ', 'inf'
	return -np.inf

##########################################
# 		The Likelihood					#
#########################################

def lnlike(theta, H0,sigH, Om,sigOm,zeq,sigZ):
	
		
	lFL3,sbar,svar,Nbar,Nvar,beta = theta
	if debugging:
		start = time.time()
		print 'in likelihood, params   ',lFL3,sbar,svar,Nbar,Nvar,beta
	# Initialise the naxion model
	# Hypervec must be correct for the model number, and match the params in theta
	# There is probably an idiot proof way to do this, but for now you have to think!
	
	my_calculator = naxion.hubble_calculator(ifsampling=True,
		fname='configuration_card_DM.ini',mnum=model,init_Kdiag=True,remove_masses=True,
		hypervec=(nax,10.**lFL3,sbar,svar,Nbar,Nvar,beta))

	###############################################################################	
	# Apply a "prior" to the log10(masses)
	# Done inside the likelihood to assure it is same random seed as for solver
	# If mass cut failed return zero likelihood, lnlik=-np.inf
	# Using remove_masses=True this should never happen.
	# Using remove_masses=False you need this for consistency.

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)
	#if debugging:
	#	print 'phivals/Mpl   ',my_calculator.phiin_array
	#	print 'log10(masses/mmax)  ',masses-mcut

	for i in range(nax):

		if masses[i]>mcut:
			if debugging:
				print 'MASSES OUTSIDE PRIOR'
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

	# Return the product likelihood for Hubble, Om, acc, zeq
	if debugging:
		print 'in likelihood, obs=   ',	Hout,Omout,add0,zout
		end = time.time()
		print 'elapsed time in  lik =   ',end-start	
		print 'lnlik = ', lnlik
	
	if derived:
		derivfile=open('Chains/'+run_name+'_derived.txt','a')
		derivfile.write("{}\t{}\t{}\t{}\t{}\n".format(Hout,Omout,add0,zout,lnlik))
		derivfile.close()	
	
	return lnlik

#############################################################################


def lnprob(theta, H0,sigH, Om,sigOm,zeq,sigZ):
	lp = lnprior(theta)
	
	if not np.isfinite(lp):
		# do not call the likelihood if the prior is infinite
		return -np.inf

	return lp + lnlike(theta, H0,sigH, Om,sigOm,zeq,sigZ)


#########################################
# 	      Do the MCMC				   #
########################################


print 'running, iteration =  ',0.,'  of  ',numiter
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm,zeq,sigZ))
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
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm,zeq,sigZ))
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
	




	