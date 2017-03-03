""" RH April 2011, 2015. Python version of the idl script of JD \n
Note the the code uses the "unweight" module -
if the CosmoMC chains are still in their weighted form. \n
You must have this module, and the optimizeFit module in the
same working directory as the code. \n

Modified by DJEM March 20167 to work with .npy files straight from emcee.
Makes more plots and titles for easy interpretation.

Original comments from Jo Dunkley's IDL code: 
Convergence test as in Dunkley et al 2004. 
Calculates the log of the power spectrum of each parameter, 
ie the FFT^2 of f_data, stored as lnPk. 
Fits lnPk with the fitting function gfunct, 
and returns the fitted log power spectra yfit_arr  
with parameters Po, alpha,jstar,r.
Plots the raw and fitted power spetra for each parameter 
Variable parameter jmax= range over which spectra are fitted. 
Should be ~10*jstar for fitting well so may have to iterate 
and check by eye the spectra have been well fit. 
For convergence require (1) that the power spectrum  
is flat at small k - check eg jstar >20.  
(2) that the sample mean variance is low: 
r<0.01 for each parameter implies variance of mean < 1% variance of distribution. \n
2015: modified to be a function rather than a script, so the modules can be called for data and chains etc. """

import numpy as np
import pylab as pl
import numpy.random as rdm
from matplotlib import rcParams
import matplotlib.pyplot as plt
import scipy.stats.mstats as stats
import optimizeFit as of
import os
rc = rcParams
rc['text.usetex'] = True
rc['font.family']='serif'
rc['font.serif'].insert(0,'Times')
rc['font.size'] = 15
rc['xtick.labelsize']='small'
rc['ytick.labelsize']='small'
rc['legend.fontsize']=15



######################################
# The function we want to fit to the data
def gfunct(x,A):
    import numpy as np
    lga = A[0]
    alpha = A[1]
    lgkp = A[2]
    z = lgkp - x # NB x = log(j)
    w = np.exp(alpha*z)
    y = lga+ np.log(w/(w+1)) #  y = lnPk
    return y

######################################

def converge(root, filename, numiter=1,burnin=1,makePlots=True):
	""" 
	root is the directory root, filename is the name of the chain
	Numiter is the number of times you want to iterate the procedure. 
	Burnin is the 1/fraction of samples to burn, so burnin=4 burns 1/4 samples.
	"""

    # your working directory
	basename = os.path.splitext(filename)[0]
	datafile = root + filename # the name of the MCMC chain
	
	mat = np.load(datafile)
	print 'data read ok'
	
	
	nwalkers, nsteps,nparams = np.shape(mat)
	

	burntot = nwalkers*nsteps/burnin
	# flatten the chain
	data=mat.reshape(nwalkers*nsteps,nparams)
	
	dataNoBurn=data.T # for chain plotting
	
	data=data[burntot:]
	
	data=data.T

	dataNorm = data # initialising the normalised vector
	nsteps = np.shape(data)[1] # the length of the array
	ndim = np.shape(data)[0] # the number of parameters
    
	#print 'nsteps (after burnin)= ', nsteps
	#print 'ndim =' , ndim
    # initialising the arrays
	ftarray = np.zeros([ndim,nsteps])
	lnPk = np.zeros([ndim,int(np.floor(nsteps/2))])
	k_data = np.zeros([ndim,int(np.floor(nsteps/2))])
    
	Po = np.zeros(ndim)
	alpha = np.zeros(ndim)
	jstar = np.zeros(ndim)
	kstar = np.zeros(ndim)
	r = np.zeros(ndim)
        
	j = np.zeros(numiter+1)
	
	# the first guess for the range, will be iterated over in the next run and taken as 10 x j*
	# DM: note that this value of j0 assumes chains of a certain minimal length
	# DM: not sure exactly what this is (from fft bookkeeping making ydat)
	# Need j[0]<nsteps/2
	jfid=999
	if nsteps/2<jfid:
		jfid = nsteps/2-1
	j[0]=jfid
	
	bias=0.577216  #Euler -Mash constant, offset for E(lnP(k)) \ne lnP(K)
	
	for dimc in range(ndim):
	
		print "               Convergence Test Results for parameter = %d"%(dimc)
		
		dataNorm = np.zeros([ndim,nsteps]) # initialise the matrix

		dataNorm[dimc,:] = stats.zmap(data[dimc,:], data[dimc,:], axis=0) 
           # normalise the general distribution into distributions 
           # with zero mean and unit variance 

		chain = dataNorm[dimc,:]

		ftarray[dimc,:] = np.fft.fft(chain)/np.float(nsteps) 
		# take the Fourier transform of the data. NB in the IDL code the fft is normalised by 1/N. 
       	
		tmplnPk = np.log((np.abs(ftarray[dimc,:])**2)*np.float(nsteps)) 
		lnPk[dimc,:] = tmplnPk[0:int(np.floor(nsteps/2))] # factor of 2 in book keeping
        # take the log of the power spectrum

###########################################
# Plotting chain
###########################################
		if makePlots:
			fig,ax1=plt.subplots()
		
			chainNoBurn=dataNoBurn[dimc,:]
			ax1.plot(chainNoBurn)
			ax1.set_title("Chain for parameter = %d"%(dimc))
			# plot the burnin amount
			x=[burntot,burntot]
			y=[chainNoBurn.min(),chainNoBurn.max()]
			ax1.plot(x,y,'k--',linewidth=2.0,alpha=0.5)
			name = root+'convergence/'+basename+'_param='+str(dimc)+'_chain.png'
			plt.savefig(name)
			plt.clf()
#############################################

		for i in range(numiter):
			ydat = lnPk[dimc,1:np.int(j[i])] # taking the log Pk so we can fit it
			xdat = np.log(np.arange(np.float(j[i])))[1:] # making a log xdat vector so we can fit it too!
			
			a_vector=[2, 2, 3]          #first-guesses for lnp0, alpha, lnj*
			coeffs = of.fit(gfunct, a_vector, ydat, xdat)
           
            
			yfit = gfunct(xdat, coeffs) + bias
			Po[dimc]=np.exp(coeffs[0]+bias)
			alpha[dimc] = coeffs[1]
			jstar[dimc] = np.exp(coeffs[2]) # should be > 20 for convergence
			kstar[dimc] = 2*np.pi/np.float(nsteps)*jstar[dimc]
			
			r[dimc] = Po[dimc]/np.float(nsteps) # convergence ratio - should be < 0.01
            
            
			if (r[dimc] < 0.01) and (jstar[dimc] > 20.0):
				conv = 'Yes'
			else:
				conv = 'No'

###############################################################################################
			print "Po = %3.2f  alpha = %5.2f  jstar = %3.2f  r = %2.2e, Converged?: %s"%(Po[dimc], alpha[dimc], jstar[dimc], r[dimc], conv) 
#############################################################################################
			if makePlots:       
				pl.plot(np.exp(lnPk[dimc,:])/np.float(nsteps))
				pl.plot(np.exp(yfit)/np.float(nsteps))
				pl.xscale('log')
				pl.yscale('log')
				pl.axis([0,np.shape(lnPk)[1], np.min(ydat)/10., np.max(ydat)*10.])
				pl.title('parameter = %d,jstar=%3.2f, r=%2.2e, Converged?=%s' %(dimc,jstar[dimc],r[dimc],conv))
				name = root+'convergence/'+basename+'_param='+str(dimc)+'_convergence_spec_j=' + repr(j[i])+'.png'
				pl.savefig(name)
				pl.clf()
		
		# DM: need j>=3 for correct number of params to constrain with brentq fit
		# number 10 here can be changed. Don't know why it is fixed.
		#if(np.floor(10.*np.min(jstar[:]))>= 4):
			j[i+1] = np.floor(10.*jstar[dimc]) 
		#else:
			#j[i+1] = 4