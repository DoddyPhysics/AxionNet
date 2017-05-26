"""
This is the workhorse of the code. Take a model and solve its equations of motion.
Compute outputs and quasiobservables.
There are some plotting scripts too.

DJEM+MS+CP+LP, 2017
"""

import scipy.integrate as integrate
import numpy as np
import numpy.random as rd

import ConfigParser
import eoms as eoms
import output as output

import model_class_v4 as model_class


class hubble_calculator(object):

	def __init__(self,fname='configuration_card.ini',ifsampling=False,mnum=None,hypervec=None,init_Kdiag=True,remove_masses=True):

		""" Initialise the object. Default uses configuration card. 
		If you use ifsampling, then mnum and hypervec are required, otherwise they are ignored.
		A thing that needs fixing here: it reads the config card on every step of an MCMC, so if you change it
		during a run, it affects the run. """
				
		myModel = model_class.ModelClass(fname=fname,ifsampling=ifsampling,init_Kdiag=init_Kdiag,remove_masses=remove_masses,
		mnum=mnum,hypervec=hypervec)
		
		# Hard code the critical density and baryon density
		self.rho_crit=3. 
		self.ombh2=0.022
		
		self.n,self.ma_array,self.phiin_array,self.phidotin_array=myModel.getParams()
		self.rho_m0,self.rhol,self.rho_r0=myModel.cosmo()
		self.ain,self.tin,self.tfi=myModel.inits()
		self.N,self.n_cross=myModel.evolution()
		# convert to integers
		self.N=int(self.N)
		self.n_cross=int(self.n_cross)
		self.n=int(self.n)
		self.crossing_index=[0]*self.n
		# use this if output = output_new
		self.crossing_array=np.zeros((self.N,self.n))
		
		
		self.rhoin_array = eoms.rhoinitial(self.phidotin_array, self.phiin_array, self.ma_array, self.n)
		self.y0 = eoms.yinitial(self.n,self.phiin_array,self.phidotin_array,self.rhoin_array,self.ain)
		
	def eq(self,y,t):
		"""Equations of motion."""
		#return eoms.deriv_wfromphi(y, t, self.n, self.n_cross,self.ma_array, self.rho_m0, self.rho_r0, self.rhol)	
		return eoms.deriv_wfromphi(y, t, self.n, self.n_cross,self.crossing_index,self.ma_array, self.rho_m0, self.rho_r0, self.rhol)	
			
	def solver(self):
		"""Solve the equations of motion with initial and final time set by class attributes."""
		
		self.t = np.logspace(np.log10(self.tin),np.log10(self.tfi),self.N)
		self.y = integrate.odeint(self.eq, self.y0, self.t, mxstep=100000000)
		
	def output(self):
		"""Obtain some derived quantities from the time steps."""
		self.rhoa = output.axionrho(self.y,self.N,self.n)
		self.rhom, self.rhor = output.dense(self.rho_m0,self.rho_r0,self.N,self.y)
		self.rholl = output.clambda(self.rhol,self.N)
		self.rhosum = output.totalrho(self.rhom, self.rholl, self.rhor, self.rhoa, self.N)
		self.P, self.Psum = output.pressure(self.y,self.ma_array,self.N,self.n,self.rhom,self.rhol,self.rhor)
		
		self.w=output.w(self.P,self.rhoa,self.N)
		self.H = output.hubble(self.t, self.rhosum)
		self.z = output.redshift(self.y, self.N)
		self.rhoo, self.rhon = output.darkflow(self.rhom, self.rhor, self.rhol, self.rhoa, self.ma_array, self.rhosum, self.n, self.y,self.N)
		self.a = output.scalefactor(self.y,self.N)
		self.add = output.accel(self.a,self.N,self.rhosum,self.Psum) 
		self.omegar, self.omegam = output.omegas(self.rhor,self.rhom,self.rhosum,self.N)
		self.phi = output.axionphi(self.y,self.N)
		self.phid = output.axionphidot(self.y,self.N)
		return self.z,self.H,self.w
	
	def phiplot(self):
		"""
		Call this to plot phis in test_sampler
		"""
		import matplotlib.pyplot as plt
		for i in range(self.n):
			plt.plot(self.t,self.y[:,3*i])
		plt.xscale('log')
		plt.show()
		
	def rhoplot(self):
		"""
		Call this to plot rhos in test_sampler
		"""
		import matplotlib.pyplot as plt
		self.rhoDMa,self.rhoDEa=output.darkflow(self.y,self.N,self.n)
		self.rhom, self.rhor = output.dense(self.rho_m0,self.rho_r0,self.N,self.y)
		self.rholl = output.clambda(self.rhol,self.N)
		self.z = output.redshift(self.y, self.N)
		inds=np.where(self.z<0)[0]

		if np.shape(inds)[0]==0:
			last=np.size(self.z)
		else:
			last=inds[0]
		self.a=output.scalefactor(self.y,self.N)		
		avec=self.a[:last]
		plt.plot(avec,self.rhoDMa[:last],'-k',linewidth=2.)
		plt.plot(avec,self.rhoDEa[:last],'--k',linewidth=2.)		
		plt.plot(avec,self.rhor[:last],'-r',linewidth=2.)
		plt.plot(avec,self.rhom[:last],'-b',linewidth=2.)
		plt.plot(avec,self.rholl[:last],'-g',linewidth=2.)
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim([1.e-5,1.e28])
		plt.show()
		
	def quasiObs(self):
		""" A very simple output of quasi-observables for MCMC """
		# First find z=0, should probably raise an exception in case this is not found
		self.z = output.redshift(self.y, self.N)
		#pos = len(self.z[self.z>=0])-1 # last positive z index
		inds=np.where(self.z<0)[0] # first negative z index
		pos=inds[0]
		if np.shape(inds)[0]==0:
			# return dummy values if z=0 is not found
			print 'z=0 was not found, returning dummy values that will fail likelihood'
			return 100.,100.,-1.,100.
		# Get all the densities and pressures for acceleration and H
		self.a=output.scalefactor(self.y,self.N)
		self.rhoa = output.axionrho(self.y,self.N,self.n)
		self.rhom, self.rhor = output.dense(self.rho_m0,self.rho_r0,self.N,self.y)
		self.rholl = output.clambda(self.rhol,self.N)
		self.rhosum = output.totalrho(self.rhom, self.rholl, self.rhor, self.rhoa, self.N)
		self.P, self.Psum = output.pressure(self.y,self.ma_array,self.N,self.n,self.rhom,self.rholl,self.rhor)
		self.H=output.hubble(self.t, self.rhosum)
		self.add = output.accel(self.a,self.N,self.rhosum,self.Psum) 
		# H0 and \ddot{a}		
		self.H0=self.H[pos]
		self.add0=self.add[pos]
		# Split the axion density into DM and DE
		self.rhoDMa,self.rhoDEa=output.darkflow(self.y,self.N,self.n)
		# Subtract the baryons to get the CDM density
		self.rhoCDM=self.rhom-self.rho_crit*self.ombh2*(1.+self.z)**3.
		self.totM=self.rhom+self.rhoDMa
		self.totDM=self.rhoCDM+self.rhoDMa
		self.totDE=self.rholl+self.rhoDEa
		# Compute Omch2 and OmM
		self.rhor0=self.rhor[pos]
		self.rhom0=self.totM[pos]
		self.rhoDE0=self.totDE[pos]
		self.Omch2=self.totDM[pos]/self.rho_crit 
		self.OmM=self.rhom0/(self.rhom0+self.rhoDE0+self.rhor0)
		# Equality
		self.zeq=output.zeq(self.z,self.totM,self.rhor)
		#return self.H0,self.OmM,self.add0,self.zeq # use this for DE models, where you care about OmM (DM is fixed)
		return self.H0,self.Omch2,self.add0,self.zeq # use this for DM models, where you care about Omch2 (Lambda is fixed)
		

############################
# Main routine runs on import (?)
# it definitely runs from the commmand line
# I don't use this for MCMC, and I run tests from test_sampler script.
##########################

#def main():

#	if len(argv)<1:
#		raise Exception('Need to specify the configuration file name via a command line argument.')
#	config_fname = argv[1]
	#Initialize calculator, which diagonalizes mass/KE matrices, etc
#	my_calculator = hubble_calculator(configname=config_fname)
	
	#Solve ODEs from tin to tfi
#	my_calculator.solver()
	
	#Save some information
#	my_calculator.output()
	
	#Make a plot
#	my_calculator.printout()
	
#if __name__ == "__main__":
#	main()
