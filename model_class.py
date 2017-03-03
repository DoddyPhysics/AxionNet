import numpy as np
import ConfigParser
import numpy.random as rd

config = ConfigParser.RawConfigParser()

class ModelClass(object):
	
	def __init__(self,fname='configuration_card.ini',ifsampling=False,mnum=None,hypervec=None):
		
		""" Initialise the object. Default uses configuration card. 
		If you use ifsampling, then mnum and hypervec are required, otherwise they are ignored. 
		This reads the theory L1 hyper parameters, and model selection. """
		
		if ifsampling:
			self.modnum = mnum
			self.hyper = np.vstack(hypervec)
			
			if self.modnum == 1:
				self.parnum = 4
			elif self.modnum == 2:
				self.parnum = 5
			elif self.modnum == 3:
				self.parnum = 5
			elif self.modnum >= 4 or self.modnum <= 0:
				raise Exception('Model number must be 1,2,3')
			
			if np.shape(self.hyper)[0]!=self.parnum:
				raise Exception('You have the wrong number of hyper parameters for your selected model')
			
			config.read(fname) # this time the config is only used for cosmo params and inits			

		else:
			config.read(fname)
			self.modnum = config.getint('Model_Selection','Model' )
			nax=config.getint('Hyperparameter','Number of Axions')

			if self.modnum == 1:
				b0 = config.getfloat('Hyperparameter','b0')
				c = config.getfloat('Hyperparameter','Dimension')
				fav = config.getfloat('Hyperparameter','fNflation')
				self.hyper=np.vstack((nax,c,b0,fav))
			elif self.modnum == 2:
				kmin = config.getfloat('Hyperparameter','kmin')
				kmax = config.getfloat('Hyperparameter','kmax')
				mmin = config.getfloat('Hyperparameter','mmin')
				mmax = config.getfloat('Hyperparameter','mmax')
				self.hyper=np.vstack((nax,kmin,kmax,mmin,mmax))
			elif self.modnum == 3:
				b0 = config.getfloat('Hyperparameter','b0')
				sb = config.getfloat('Hyperparameter','sigma_b')
				s1 = config.getfloat('Hyperparameter','s1')
				s2 = config.getfloat('Hyperparameter','s2')
				self.hyper=np.vstack((nax,s1,s2,b0,sb))

	
		

		
#######################################################	
	def getParams(self):
		"""" Return the number of axions, masses and phivals, this is theory L1 and also reads the phi range """
		###################################################################
		###################################################################
		####                        Models                             ####
		###################################################################
		###################################################################

		mo=self.modnum
		n=int(self.hyper[0])

		###################################################################
		####              Easther - McAllister Model (1)                ####
		###################################################################

		
		if mo == 1:
			# hyper is (c,sb,fav)
			c=self.hyper[1]
			b0=self.hyper[2]
			fav=self.hyper[3]
			fav=fav[0]
			
			######################################
			####          Kahler, trivial for this model, but we go through the motions anyway         ####
			######################################		
		
			kk = np.empty((n,n))
			kk.fill(1) 
			#kk = np.full((n, 1), 1) 
			kk3=np.diag(kk[:,0])
			kT = kk3.transpose() # transpose of random matrix k
			k2 = np.dot(kk3,kT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,pT = np.linalg.eigh(k2) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)*fav # This is an implicit choice for f in this model
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kD = reduce(np.dot, [p, k2, pT]) #diagonalisation of Kahler metric
			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/((2**0.5)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
			#kDr[kDr > 1*10**23] = 0 # remove computational errors in reciprocal matrix
			kDrT = kDr.transpose() # trasnpose of kDr matrix

			######################################
			####            Mass              ####
			######################################
		
		
			L=int(n/c)
			X = b0*(np.random.randn(n, L)) 
			Wc = np.dot(X,(X.T))/L
			mn = reduce(np.dot, [pT,kDrT, Wc, kDr,p]) # correct mass matrix caclulation
			ma_array,mv = np.linalg.eigh(mn) # reout of masses^2 from eigenvalues of mn
			ma_array = np.sqrt(ma_array)
		
		####################################################################
		####################################################################


		###################################################################
		####                     LogFlat Model (2)                     ####
		###################################################################

		if mo == 2:
			# hyper is a0,sa,mlow,mup
			kmin=self.hyper[1]
			kmax=self.hyper[2]
			mmin=self.hyper[3]
			mmax=self.hyper[4]

			######################################
			####          Kahler              ####
			######################################

			k = (np.random.uniform(kmin,kmax,(n,n))) #random matrix k from log normal distribution
			kk = np.exp(-k)
			kT = kk.transpose() # transpose of random matrix k
			k2 = np.dot(kk,kT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,pT = np.linalg.eigh(k2) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)
			fef2=np.log(fef)
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kD = reduce(np.dot, [p, k2, pT]) #diagonalisation of Kahler metric
			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/((2**0.5)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
			#kDr[kDr > 1*10**23] = 0 # remove computational errors in reciprocal matrix
			kDrT = kDr.transpose() # trasnpos

			######################################
			####            Mass              ####
			######################################

			m = (np.random.uniform(mmin,mmax,(n,n))) #random matrix m from log flat
			mm = np.exp(-m)
			mT = mm.transpose() # transpose of random matrix m
			m2 = np.dot(mm,mT) # symmeterised mass matrix from real axion fields
			mn = reduce(np.dot, [pT,kDrT, m2, kDr,p]) # correct mass matrix caclulation
			ma_array,mv = np.linalg.eigh(mn) # reout of masses^2 from eigenvalues of mn
			ma_array = np.sqrt(ma_array)
			
		####################################################################
		####################################################################

		###################################################################
		####                     Bobby's Model (3)                     ####
		###################################################################

		if mo == 3:
			# hyper is s1,s2,b0,sb
			s1=self.hyper[1]
			s2=self.hyper[2]
			b0=self.hyper[3]
			sb=self.hyper[4]
			# I am setting a0 to 1 here: I think there are implicit units!
			a0=1.

			######################################
			####          Kahler              ####
			######################################

			k = (a0/np.random.uniform(s1,s2,(n,n))) #random matrix k from log normal distribution
			kT = k.transpose() # transpose of random matrix k
			k2 = np.dot(k,kT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,pT = np.linalg.eigh(k2) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kD = reduce(np.dot, [p, k2, pT]) #diagonalisation of Kahler metric
			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/((2**0.5)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
			#kDr[kDr > 1*10**23] = 0 # remove computational errors in reciprocal matrix
			kDrT = kDr.transpose() # trasnpose of kDr matrix

			######################################
			####            Mass              ####
			######################################

			m = (np.random.uniform(np.log(b0)-sb,np.log(b0)+sb,(n,n))) #random matrix m from log normal distribution
			mm = np.exp(-m)
			mT = mm.transpose() # transpose of random matrix m
			m2 = np.dot(mm,mT) # symmeterised mass matrix from real axion fields
			mn = reduce(np.dot, [pT,kDrT, m2, kDr,p]) # correct mass matrix caclulation
			ma_array,mv = np.linalg.eigh(mn) # reout of masses^2 from eigenvalues of mn
			ma_array = np.sqrt(ma_array)
			
		phi_range = config.getfloat('Initial Conditions','phi_in_range')
		phidotin = config.getfloat('Initial Conditions','phidot_in')
		phiin_array = rd.uniform(0.,phi_range,n)

		for i in range (0,n):
			phiin_array[i] = phiin_array[i]*fef[i]
		phiin_array=np.dot(mv,phiin_array)
		phidotin_array = [phidotin]*n #### array of phidotin where all are set equal to zero
		
		return n,ma_array,phiin_array,phidotin_array
		
				
		
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