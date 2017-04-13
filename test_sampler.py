"""
Do some test sampling from a model.
This is a bit messy, as I just hack it every time.
You might find some useful stuff here though.

DJEM, 2017
"""

import naxion
import numpy as np
import model_class
import matplotlib.pyplot as plt
from quasi_observable_data import *


import time
np.random.seed(121)

numsamps=10

# Mtheory
model=3
nax=20
#fval=10**(-1.47) 
#beta=0.78
#b0=10**(7.4)
lFL3=105.
#lL=0.
smin=25.
smax=50.
N=0.7
betaM=1.


# MP-DE
#beta=0.5
#b0=0.2
#f=0.3

# MP-DM
beta=0.5
b0=10**(7.)
f=10**(-1.3)

#myModel = model_class.ModelClass(ifsampling=True,mnum=model,hypervec=(nax,beta,b0,fval))
#myModel = model_class.ModelClass(fname='configuration_card.ini')

#n,ma_array,phiin_array,phidotin_array=myModel.getParams()
#print phiin_array
#print ma_array
#rhoin_array = eoms.rhoinitial(phidotin_array, phiin_array, ma_array, n)
#print rhoin_array

#my_calculator = naxion.hubble_calculator(fname='configuration_card_temp.ini',ifsampling=True,mnum=model,hypervec=(nax,beta,b0,fval))
#my_calculator.solver()

#z,H,phi=my_calculator.output()

#plt.plot(z,np.abs(phi))
#plt.xscale('log')
#plt.yscale('log')
#plt.show()

# Check that for small axion density (i.e. small fval=1.e-2 in N-flation model)
# we get cosmology close to LCDM with h~0.68 and the given matter content
# omh2=0.32*0.68**2.=0.148, olh2=0.68*0.68**2.=0.314

for i in range(numsamps):
	#start=time.time()
	#print 'computing sample=',i,'...'

# Mtheory-DM
	my_calculator = naxion.hubble_calculator(ifsampling=True,fname='configuration_card_DM.ini',mnum=model,
		hypervec=(nax,10**lFL3,smin,smax,N,betaM),init_Kdiag=True,remove_masses=False)

# MP
#	my_calculator = naxion.hubble_calculator(ifsampling=True,fname='configuration_card_DM.ini',mnum=1,
#		hypervec=(nax,beta,b0,f),init_Kdiag=True,remove_masses=False)

	masses=my_calculator.ma_array*MH
	masses=np.log10(masses)
	print 'masses=  ',masses
	#if debugging:
	#	print 'phivals/Mpl   ',my_calculator.phiin_array
	#	print 'log10(masses/mmax)  ',masses-mcut
	if np.logical_not(masses[masses-mcut>0].size):
	#for i in range(nax):
	#	if masses[i]>mcut:
	#		print 'MASSES OUTSIDE PRIOR'
			
		my_calculator.solver()
		Hout,Omout,add0,zeq=my_calculator.quasiObs()
	#my_calculator.phiplot()
		my_calculator.rhoplot()	
		print 'outputs=',Hout,Omout,add0,zeq
	else:
		print 'masses failed cut'
	
	
	
	#end=time.time()
	#print 'time =   ',end-start
	#z,H,rhosum=my_calculator.output()
	#plt.plot(z,H)
	#dat=np.vstack((z,rhosum))
	#np.savetxt('TestOutputs/test_out_mod_'+str(i)+'.txt',dat.T)
	#plt.plot(z,rhosum)

#hvec=np.ones(len(z))
#plt.plot(z,hvec*0.68,'-k')
#plt.xlim([0,10])
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim([1.e-2,1.e3])
#plt.ylim([0,10])
#plt.ylim([-1.,1.])
#plt.show()

