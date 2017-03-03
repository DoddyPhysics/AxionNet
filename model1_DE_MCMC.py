import emcee
import numpy as np
import math
import naxion
from quasi_observable_data import *

##############################
# 	Set up your model		 #
##############################

# You must be careful here selecting the right 
# parameters for the model, and priors, and match them up  
# in the prior and likelihood functions below.
# This is not idiot proof!

# Model selection
run_name='model1_nax20_DE_run1'
nax=20
model=1

# Sampler parameters
# ndim must be correct for the model!
ndim, nwalkers, nsteps = 3, 10, 5000

# Priors
fmin,fmax=0.,5.e0
betamin,betamax=0.,1.
b0min,b0max=0.,1.e1

# Starting position
startFile=False
if startFile:
	startChain=np.load('Chains/model1_nax20_DE_run1.npy')
	# Take last sample from each walker as new starting position
	pos=startChain[:,-1,:]
else:
	# Uniform starts
	# Match these to the prior if using e.g. log-flat
	pos = [[np.random.uniform(fmin,fmax),np.random.uniform(betamin,betamax),np.random.uniform(b0min,b0max)] for i in range(nwalkers)]

##############################
# Likelihood and prior functions
##############################

def lnprior(theta):
	fval,beta,b0 = theta
		
	# Flat priors on parameters
	if fmin<fval<fmax and betamin<beta<betamax and b0min<b0<b0max:
		return 0.0
	return -np.inf

##########################################
# 		The Likelihood					#
#########################################

def lnlike(theta, H0,sigH, Om,sigOm):
		
	fval,beta,b0 = theta

	# Initialise the naxion model
	# Hypervec must be correct for the model number, and match the params in theta
	# There is probably an idiot proof way to do this, but for now you have to think!
	my_calculator = naxion.hubble_calculator(ifsampling=True,fname='configuration_card_DE.ini',mnum=model,hypervec=(nax,beta,b0,fval))

	###############################################################################	
	# Apply a "prior" to the log10(masses)
	# Done inside the likelihood to assure it is same random seed as for solver
	# If mass cut failed return zero likelihood, lnlik=-np.inf

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

	#############################################################################

	# Return the product likelihood for Hubble, Om, acc, zeq	
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

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(H0,sigH, Om,sigOm))

# Run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
print("Done.")

#####################################
# Saving chains
#####################################

# I am saving the chains at the end, not saving the state during a run.
# Could set up a loop and save every fixed number of steps

print("Saving Chains...")
np.save('Chains/'+run_name+'.npy',sampler.chain[:,:,:])

