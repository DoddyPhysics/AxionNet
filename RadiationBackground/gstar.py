"""
gstarS and gstarR fits from Wantz and Shellard, 0910.1066, Appendix A

"""

import numpy as np

a0S=1.36

a1S=np.asarray([0.498,0.327,0.579,0.140,0.109])
a2S=np.asarray([-8.74,-2.89,-1.79,-0.102,3.82])
a3S=np.asarray([0.693,1.01,0.155,0.963,0.907])

def gS(T):
	"""
	The input temperature is measured in eV
	gstarS as a function of T from fits
	"""
	T=T/1.e9
	t=np.log(T)
	f=a0S
	for i in range(0,5):
		f=f+a1S[i]*(1.+np.tanh((t-a2S[i])/a3S[i]))
	return np.exp(f)

a0R=1.21
a1R=np.asarray([0.572,0.330,0.579,0.138,0.108])
a2R=np.asarray([-8.77,-2.95,-1.80,-0.162,3.76])
a3R=np.asarray([0.682,1.01,0.165,0.934,0.869])

def gR(T):
	"""
	The input temperature is measured in eV
	gstarR as a function of T from fits
	"""
	T=T/1.e9
	t=np.log(T)
	f=a0R
	for i in range(0,5):
		f=f+a1R[i]*(1.+np.tanh((t-a2R[i])/a3R[i]))
	return np.exp(f)


#import matplotlib.pyplot as plt


#T=np.logspace(-6,3,100)
#plt.plot(T,gS(T),linewidth=2.0)
#plt.plot(T,gR(T),'-r',linewidth=2.0)
#plt.ylim([1.,200.])
#plt.xscale('log')
#plt.yscale('log')
#plt.show()

		

