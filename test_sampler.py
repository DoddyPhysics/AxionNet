import naxion
import numpy as np
import model_class
import matplotlib.pyplot as plt

#np.random.seed(121)

model=1
nax=20
fval=1.e-2 
beta=0.5
b0=1.e8

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

for i in range(0,10):
	print 'computing sample=',i,'...'
	my_calculator = naxion.hubble_calculator(ifsampling=True,fname='configuration_card_DM.ini',mnum=model,hypervec=(nax,beta,b0,fval))
	my_calculator.solver()
	Hout,Omout,add0,zeq=my_calculator.quasiObs()
	#my_calculator.phiplot()
	print 'sample=',i,'outputs=',Hout,Omout,add0,zeq
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

