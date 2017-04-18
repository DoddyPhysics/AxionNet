"""
This is the theory part of the code. 
Compute the mass and initial conditions spectrum.
Get all other params.

DJEM+MS+CP, 2017
"""

import numpy as np
import ConfigParser
import numpy.random as rd
import quasi_observable_data as quasi # for mcut

config = ConfigParser.RawConfigParser()

# Note that np.linalg.eig(M) returns the normalized eigenvectors as an array with eigenvectors as columns.
# The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]. 
# i.e. it returns the (right acting) rotation matrix.
# If eigval,eigvec=np.linalg.eig(M), then np.dot(eigvec.T,np.dot(M,eigvec)) is the diagonal matrix with eigval entries
# and rotating vectors goes as np.dot(eigv,x) = R.x = R_{ij} x_j

class ModelClass(object):
	
	"""
	Functions are:
	getParams: returns axion model params (n,m,phii,phidoti)
	inits: returns integrator conditions (ai,ti,tf)
	evolution: accuracy parameters (nsteps,ncross)
	cosmo: cosmo parameters (Omh2,Olh2,Orh2)
	"""
	
	def __init__(self,fname='configuration_card.ini',ifsampling=False,remove_masses=True,init_Kdiag=True,
		mnum=None,hypervec=None):
		
		""" 
		Initialise the object. This reads the theory L1 hyper parameters, and model selection. 
		Default uses configuration card. 
		ifsampling: use mnum and hypervec as entries, otherwise they are ignored. 
		remove_masses: how to treat masses outside the allowed range . Default: set to zero as "decayed".
		init_Kdiag: How to set i.c.'s. Default is square theta lattice in K-diag basis.
		"""
		
		self.init=init_Kdiag
		self.remove=remove_masses
		
		config.read(fname)
		if ifsampling:
			# the config card is only used for cosmo params and inits
			self.modnum = mnum
			self.hyper = np.vstack(hypervec)
			
			if self.modnum == 1:
				self.parnum = 4
			elif self.modnum == 2:
				self.parnum = 7
			elif self.modnum == 3:
				self.parnum = 6
			elif self.modnum == 4:
				self.parnum = 5
			elif self.modnum == 5:
				self.parnum = 5		
			elif self.modnum >= 6 or self.modnum <= 0:
				raise Exception('Model number must be 1,2,3,4,5')
			
			if np.shape(self.hyper)[0]!=self.parnum:
				raise Exception('You have the wrong number of hyper parameters for your selected model')
			

		else:
			# Read axion params from config_card, make into hypervec
			
			self.modnum = config.getint('Model_Selection','Model' )
			nax=config.getint('Hyperparameter','Number of Axions')

			if self.modnum == 1:
				b0 = config.getfloat('Hyperparameter','b0')
				betaM = config.getfloat('Hyperparameter','Dimension')
				fav = config.getfloat('Hyperparameter','fNflation')
				self.hyper=np.vstack((nax,betaM,b0,fav))
			elif self.modnum == 2:
				betaK = config.getfloat('Hyperparameter','betaK')
				betaM = config.getfloat('Hyperparameter','betaM')				
				kmin = config.getfloat('Hyperparameter','kmin')
				kmax = config.getfloat('Hyperparameter','kmax')
				mmin = config.getfloat('Hyperparameter','mmin')
				mmax = config.getfloat('Hyperparameter','mmax')
				self.hyper=np.vstack((nax,betaK,betaM,kmin,kmax,mmin,mmax))
			elif self.modnum == 3:
				F = config.getfloat('Hyperparameter','F')
				Lambda = config.getfloat('Hyperparameter','Lambda')
				smin = config.getfloat('Hyperparameter','smin')
				smax = config.getfloat('Hyperparameter','smax')
				Ntildemax = config.getfloat('Hyperparameter','Ntildemax')
				betaM = config.getfloat('Hyperparameter','betaM')
				self.hyper=np.vstack((nax,F,Lambda,smin,smax,Ntildemax,betaM))
			elif self.modnum == 4:
				betaK = config.getfloat('Hyperparameter','betaK')
				betaM = config.getfloat('Hyperparameter','betaM')
				a0 = config.getfloat('Hyperparameter','a0')
				b0 = config.getfloat('Hyperparameter','b0')
				self.hyper=np.vstack((nax,betaK,betaM,a0,b0))	
			elif self.modnum == 5:
				kmin = config.getfloat('Hyperparameter','kmin')
				kmax = config.getfloat('Hyperparameter','kmax')
				mmin = config.getfloat('Hyperparameter','mmin')
				mmax = config.getfloat('Hyperparameter','mmax')
				self.hyper=np.vstack((nax,kmin,kmax,mmin,mmax))	

#########################################
# Chakrit functions
########################################

	def poscheck(ev):
		"""
		This is to flag negative eigenvalues and exit, we don't use it anywhere but you can.
		"""
		if any(x <= 0 for x in ev):
			raise Exception('You have negative eigenvalues')
		else:
			return 0

	def checkvolume(n,smin,smax,Imax,Nvolmax,Idist):

		s = np.random.uniform(smin,smax,n)
		##### 1 = uniform distribution of bij from [0,Imax - 1]
		##### 2 = Poisson distribution of bij with lambda = Imax
		##### 3 = 'diagonal' uniform distribution of bij from [0,Imax - 1]
		if Idist == 1:
			Nvol = 2*np.pi*np.random.poisson(Nvolmax,size=(n,n))
		elif Idist == 2:
			Nvol = 2*np.pi*np.random.poisson(Nvolmax,size=(n,n))
		else:
			Nvol = np.zeros((n, n))
			#np.fill_diagonal(b, 2*np.pi*np.random.randint(Imax,size=n))
			np.fill_diagonal(Nvol, 2*np.pi*np.random.poisson(Nvolmax,size=n))
		return np.dot(s,Nvol)
				
#######################################################
# The main function gets model parameters
######################################################

	def getParams(self):
		
		"""" Return the number of axions, masses and phivals, i.e. the model parameters. 
		This converts theory L1 to model params. Reads theta range. """

		###################################################################
		###################################################################
		####                        Models                             ####
		###################################################################
		###################################################################

		mo=self.modnum
		n=int(self.hyper[0])

		###################################################################
		####              Easther - McAllister Model (1)               ####
		###################################################################

		
		if mo == 1:
			
			betaM=self.hyper[1]
			b0=self.hyper[2]
			fav=self.hyper[3]
			
			###############################################################################################	
			####          Kahler, trivial for this model, but we go through the motions anyway         ####
			###############################################################################################	
				
			kk = np.empty((n,n))
			kk.fill(1.) 
			kk2=np.diag(kk[:,0])
			kkT = kk2.transpose() # transpose of random matrix k
			k2 = np.dot(kk2,kkT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,p = np.linalg.eig(k2) # calculation of eigen values and eigen vectors
			#ev,p = np.linalg.eig(kk) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)*fav # This is an implicit choice for f in this model
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)
#			kD = reduce(np.dot, [p.T, k2, p]) #diagonalisation of Kahler metric
#			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))
			np.fill_diagonal(kDr, (1./(fef))) # matrix for absolving eigen values of kahler metric into axion fields
			
			######################################
			####            Mass              ####
			######################################
		
			L=int(n/betaM)
			X = b0*(np.random.randn(n, L)) 
			M = np.dot(X,(X.T))/L # why this factor of L?? Can't find justification....
			mn = 2.*reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
			ma_array,mv = np.linalg.eig(mn) # reout of masses^2 from eigenvalues of mn
			ma_array = np.sqrt(ma_array)
		
		####################################################################
		####################################################################


		###################################################################
		####               Log-Flat Elements Model (2)                 ####
		###################################################################

		if mo == 2:
			# hyper is a0,sa,mlow,mup
			betaM=self.hyper[1]
			betaK=self.hyper[2]
			kmin=self.hyper[3]
			kmax=self.hyper[4]
			mmin=self.hyper[5]
			mmax=self.hyper[6]

			######################################
			####          Kahler              ####
			######################################
			LK=int(n/betaK)
			LM=int(n/betaM)
			k = (np.random.uniform(kmin,kmax,(n,LK))) #random matrix k from log flat distribution
			k = 10.**k
			k2 = np.dot(k,k.T)/LK # Factor of L  
			ev,p = np.linalg.eig(k2) 
			fef = np.sqrt(2.*ev)
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)
#			kD = reduce(np.dot, [p.T, k2, p]) #diagonalisation of Kahler metric
#			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))
			np.fill_diagonal(kDr, (1./(fef)))

			######################################
			####            Mass              ####
			######################################

			m = (np.random.uniform(mmin,mmax,(n,LM))) #random matrix m from log flat
			m = 10.**m
			m2 = np.dot(m,m.T) /LM # Factor of L
			mn = 2.*reduce(np.dot, [kDr,p,m2,p.T,kDr.T]) 
			ma_array,mv = np.linalg.eig(mn) 
			ma_array = np.sqrt(ma_array)
			
		####################################################################
		####################################################################

		###################################################################
		####                       M-theory (3)                        ####
		###################################################################

		if mo == 3:
			
			# DM: since F and Lambda appear in exactly the same way, I am 
			# reducing the number of params and sampling only in (FL^3)
			# Setting L=1 and F=m_{3/2}M_{pl}/M_H^2
			FL3=self.hyper[1]
			#Lambda = self.hyper[2]
			smin=self.hyper[2]
			smax=self.hyper[3]
			Ntildemax=self.hyper[4]
			betaM=self.hyper[5]
		
			# I am setting a0 to 1 here: I think there are implicit units!
			a0=1.

			remove_tachyons=True
			######################################
			####          Kahler              ####
			######################################
		
			s = np.random.uniform(smin,smax,n)
			
			###############################
			# General tensor dot case
			#k = np.tensordot(a0/s,a0/s,axes=0) # This is not strictly positive definite!!
			###############################
			k = np.zeros((n,n))
			np.fill_diagonal(k,a0*a0/s/s)
			ev,p = np.linalg.eig(k) # calculation of eigenvalues and eigenvectors
			fef = np.sqrt(np.abs(2.*ev))
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)
			kDr = np.zeros((n, n))
			np.fill_diagonal(kDr, (1./(fef)))
					
			######################################
			####            Mass              ####
			######################################
			
			#if Idist == 1:
				#b = [2*np.pi*np.random.randint(Imax,size=n)]
			#	b = [1]*n
			#	Ntilde = np.random.poisson(Ntildemax,size=(n,n))
			#elif Idist == 2:
				#b = 2*np.pi*np.random.poisson(Imax,size=n)
			#	b = [1]*n
			#	Ntilde = np.random.uniform(0,Ntildemax,size=(n,n))			
			#else:
			#	b = [1]*n
			#	Ntilde = np.zeros((n, n))
				#np.fill_diagonal(b, 2*np.pi*np.random.randint(Imax,size=n))
			#	np.fill_diagonal(Ntilde, 2*np.pi*np.random.randint(Ntildemax,size=n))
				
			##########################
			
			L = int(n/betaM)
			Ntilde = np.random.uniform(0,Ntildemax,size=(n,L))			
			Sint = np.dot(s,Ntilde)
			Esint = np.exp(-Sint/2.)
			Idar = n*[1.]
			Cb = np.sqrt(np.dot(Idar,Ntilde))
					
			#A = 2.*np.sqrt(F*Lambda*Lambda*Lambda)*reduce(np.multiply,[Cb,Esint,Ntilde]) 
			A = 2.*np.sqrt(FL3)*reduce(np.multiply,[Cb,Esint,Ntilde]) 

			m = np.dot(A,A.T)/L # factor of L as for Marchenko-Pastur. Correct?
			mn = 2.*reduce(np.dot, [kDr,p,m,p.T,kDr.T]) 
			ma_array,mv = np.linalg.eigh(mn) 
			
			######################################
			# Note on eigenvalues
			######################################
			# We use eigh here: numerical error is making mn non-symmetric, even though it is mathematically symmetric
			# eigh assumes Hermitian and returns real eigenvectors. 
			# Still have a problem of tachyons from numerical error. This is only for the lightest masses.
			# Using "remove tachyons" removes them from spectrum, otherwise we use abs and assume they are positive.
			# This is caused by the large spread in eigenvalues in the Mtheory model.
			# See note in stack overflow: http://stackoverflow.com/questions/36819739/scipy-eigh-gives-negative-eigenvalues-for-positive-semidefinite-matrix
			# Try this to do better, but we may be screwed:
			# http://stackoverflow.com/questions/6876377/numpy-arbitrary-precision-linear-algebra
			#####################################
			#print ma_array, 'bare masses'
				
			# remove any tachyons by setting to zero "as if decayed"
			if remove_tachyons:
				tachyons=ma_array[ma_array<0]
				print np.shape(tachyons), 'number of tachyons'
				ma_array[ma_array<0]=0.
			
			ma_array = np.sqrt(np.abs(ma_array))



		####################################################################
		####################################################################
		
		###################################################################
		####               Wishart/Wishart Model   (4)                 ####
		###################################################################

		if mo == 4:
			
			betaM=self.hyper[1]
			betaK=self.hyper[2]
			a0=self.hyper[3]
			b0=self.hyper[4]
			
			######################################
			####          Kahler              ####
			######################################

			LK=int(n/betaK)
			LM=int(n/betaM)
			k  = a0*(np.random.randn(n, LK))
			k2 = np.dot(k,(k.T))/LK # Factor of L
			ev,p = np.linalg.eig(k2) 
			fef = np.sqrt(np.abs(2.*ev))
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)	 
			kD = reduce(np.dot, [pT, k2, p]) 
			kD[kD < 1*10**-13] = 0 
			kDr = np.zeros((n, n)) 
			np.fill_diagonal(kDr, 1./(fef))
			
			######################################
			####            Mass              ####
			######################################
			
			m = b0*(np.random.randn(n, LM)) 
			M = np.dot(m,(m.T))/LM # Factor of L
			mn = 2.*reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
			eigs,mv = np.linalg.eig(mn) 
			ma_array=np.sqrt(eigs)
			
		####################################################################
		####################################################################	
			
		###################################################################
		####                  Haar Measure Model   (5)                 ####
		###################################################################

		if mo == 5:

			kmin=self.hyper[1]
			kmax=self.hyper[2]
			mmin=self.hyper[3]
			mmax=self.hyper[4]
			
			######################################
			####          Kahler              ####
			######################################
	
			k = (np.random.uniform(kmin,kmax,(n))) 
			k = 10.**k 
			fef=np.sqrt(2.*k)
			fmat = np.zeros((n,n))
			np.fill_diagonal(fmat,fef)
			p=1.
					
			######################################
			####            Mass              ####
			######################################
			
			m = (np.random.uniform(mmin,mmax,(n))) 
			m = 10.**m
			ma_array=np.sqrt(2.*m)
			mv=1.
	
				
			####################################################################
			####################################################################	
			
		phi_range = config.getfloat('Initial Conditions','phi_in_range')
		phidotin = config.getfloat('Initial Conditions','phidot_in')
		phiin_array = rd.uniform(0.,phi_range,n)

		if self.init:
			# Set initial conditions as cubic lattice in basis where K is diag.
			phiin_array=reduce(np.dot,[mv,fmat,phiin_array])
		else:
			# General basis is cubic lattice.
			phiin_array=reduce(np.dot,[mv,fmat,p,phiin_array])
		
		phidotin_array = [phidotin]*n 
		
		if self.remove:
			# Set masses that fail the cut to be zero.
			# This is "as if those axions decayed", because our i.c.'s remove them from the spectrum.
			#cut_masses=ma_array[np.log10(ma_array*quasi.MH)-quasi.mcut>0.]
			#print np.shape(cut_masses), 'cut mass'
			ma_array[np.log10(ma_array*quasi.MH)-quasi.mcut>0.]=0.
			
		
		return n,ma_array,phiin_array,phidotin_array

########################################
# Read and return non-axion parameters
#######################################		
				
	def inits(self):
		""" Initial conditions"""
				
		ain = config.getfloat('Initial Conditions','a_in')
		tin = config.getfloat('Initial Conditions','t_in')
		tfi = config.getfloat('Initial Conditions','t_fi')
		
		return ain,tin,tfi

	def evolution(self):
		""" Evolution Settings """
		nsteps = config.getfloat('Evolution Settings','Number of time steps')
		ncross = config.getfloat('Evolution Settings','Number of Crossings')
		
		return nsteps,ncross
	
	def cosmo(self,ifsampling=False):
		""" Return the common vector of cosmo params """
		rhocrit=3. # units h**2.Mpl**2.*MH**2.
		rho_mat = config.getfloat('Cosmo Params','Omh2')*rhocrit
		rho_lam = config.getfloat('Cosmo Params','Olh2')*rhocrit
		rho_rad = config.getfloat('Cosmo Params','Orh2')*rhocrit
		
		return rho_mat,rho_lam,rho_rad
			
	####################################################################
	####################################################################