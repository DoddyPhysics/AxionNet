"""
Compute statistics from your chain.

DJEM, 2017
"""

import numpy as np
import matplotlib.pyplot as plt

run_name='Mtheory_wide_nax20_DM_run1'

############################
# Settings
###########################

modes=True # modes were slow with scipy modes, fast with histogram
thin=False # thin the chain
run=True # compute the stats
hist=True # Look at histograms if you want a sanity check

#############################
# Initialising
############################

chain=np.load('Chains/'+run_name+'.npy')
nwalkers, nsteps,ndim = np.shape(chain)
burnin = nsteps/4
# Make sample chain removing burnin 
chain=chain[:,burnin:,:].reshape((-1,ndim))
nsteps=np.shape(chain)[0]

thinpar=0.01 # factor to thin chain by
binnum=20 # bins for histogram and mode

############################
# Computing stats
############################



for i in range(ndim):
	if thin:
		newsteps=int(nsteps*thinpar)
		param = np.random.choice(param,newsteps)
	if run:
		param=chain[:,i]
	
		print 'param number   ', i
		print 'mean=', np.mean(param), '   stdev=',  np.std(param)
		print 'median= ', np.percentile(param,[50.])
		print  'lower 95,  upper 95   ', np.percentile(param,[5.,95.])
		if modes:
			# we bin the data for modes
			counts,edges=np.histogram(param,bins=binnum)
			ind=np.argmax(counts,axis=0)
			val=(edges[ind+1]+edges[ind])/2.
			print 'mode=   ', val
	if hist:
		plt.hist(param,bins=binnum)
		plt.show()
			
		