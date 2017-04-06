import numpy as np
import ConfigParser
import numpy.random as rd

config = ConfigParser.RawConfigParser()

# Note that np.linalg.eig(M) returns the normalized eigenvectors as an array with eigenvectors as columns.
# The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]. 
# i.e. it returns the (right acting) rotation matrix.
# If eigval,eigvec=np.linalg.eig(M), then np.dot(eigvec.T,np.dot(M,eigvec)) is the diagonal matrix with eigval entries
# and rotating vectors goes as np.dot(eigv,x) = R.x = R_{ij} x_j

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
				self.parnum = 6
			elif self.modnum == 4:
				self.parnum = 4
			elif self.modnum == 5:
				self.parnum = 5		
			elif self.modnum >= 6 or self.modnum <= 0:
				raise Exception('Model number must be 1,2,3,4,5')
			
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
				F = config.getfloat('Hyperparameter','F')
				Lambda = config.getfloat('Hyperparameter','Lambda')
				smin = config.getfloat('Hyperparameter','smin')
				smax = config.getfloat('Hyperparameter','smax')
				Ntildemax = config.getfloat('Hyperparameter','Ntildemax')
				#flag = config.getint('Hyperparameter','Flag')
				#a0 = config.getfloat('Hyperparameter','a0')
				self.hyper=np.vstack((nax,F,Lambda,smin,smax,Ntildemax))
			elif self.modnum == 4:
				c = config.getfloat('Hyperparameter','Dimension')
				a0 = config.getfloat('Hyperparameter','a0')
				b0 = config.getfloat('Hyperparameter','b0')
				self.hyper=np.vstack((nax,c,a0,b0))	
			elif self.modnum == 5:
				kmin = config.getfloat('Hyperparameter','kmin')
				kmax = config.getfloat('Hyperparameter','kmax')
				mmin = config.getfloat('Hyperparameter','mmin')
				mmax = config.getfloat('Hyperparameter','mmax')
				self.hyper=np.vstack((nax,kmin,kmax,mmin,mmax))	

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
				
#######################################################	#######################################################	

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
		####              Easther - McAllister Model (1)               ####
		###################################################################

		
		if mo == 1:
			# hyper is (c,sb,fav)
			c=self.hyper[1]
			b0=self.hyper[2]
			fav=self.hyper[3]
			fav=fav[0]
			
			###############################################################################################	
			####          Kahler, trivial for this model, but we go through the motions anyway         ####
			###############################################################################################	
		
		
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
			np.fill_diagonal(kDr, (1/((1)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
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
		####               Log-Flat Elements Model (2)                 ####
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
			fef = np.sqrt(2*ev)
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
		####                       M-theory (3)                        ####
		###################################################################

		if mo == 3:

			F=self.hyper[1]
			Lambda = self.hyper[2]
			smin=self.hyper[3]
			smax=self.hyper[4]
			Ntildemax=self.hyper[5]
			#flag = self.hyper[6]
			#a0=self.hyper[7]

			# I am setting a0 to 1 here: I think there are implicit units!
			a0=1.

		######################################
		####          Kahler              ####
		######################################
		
			s = np.random.uniform(smin,smax,n)
			#s = np.random.randn(smax,n)
			#k = np.tensordot(1/s,1/s,axes=0) # This is not strictly positive definite!!
			k = np.zeros((n,n))
			np.fill_diagonal(k,a0*a0/s/s)
			ev,pT = np.linalg.eig(k) # calculation of eigen values and eigen vectors
			fef = np.sqrt(np.abs(2.*ev))
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/fef))# matrix for absolving eigen values of kahler metric into axion fields

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
			b = [1]*n # instanton charges
			Ntilde = np.zeros((n, n)) 
			np.fill_diagonal(Ntilde, 2*np.pi*np.random.randint(Ntildemax,size=n)) # Assume diagonal N in gauge kinetic
				
			##########################
			
			Sint = np.multiply(b,np.dot(Ntilde,s))
			Esint = np.exp(-Sint/2)
			Idar = n*[1]
			Cb = np.multiply(np.dot(Ntilde,Idar),b)
			A = np.sqrt(F*Lambda*Lambda*Lambda)*reduce(np.multiply,[np.sqrt(Cb),Esint,b,np.transpose(Ntilde)])
			AT = np.transpose(A)
			m = np.dot(A,AT)
			mn = reduce(np.dot, [pT,kDr, m, kDr,p]) # correct mass matrix calculation
			ma_array2,mv = np.linalg.eig(mn)
			#flag = poscheck(ma_array2)
			ma_array = np.sqrt(np.abs(ma_array2))
			#### The chance of having negative eigenvalues might crash the code, 
			###  We do sqrt thing outside
			###  So that we can check and discard any bad samples....


		####################################################################
		####################################################################
		
		###################################################################
		####               Wishart/Wishart Model   (4)                 ####
		###################################################################

		if mo == 4:
			
			c=self.hyper[1]
			a0=self.hyper[2]
			b0=self.hyper[3]
			
			######################################
			####          Kahler              ####
			######################################

			L=int(n/c)
			K  = a0*(np.random.randn(n, L))
			Kc = np.dot(K,(K.T))/L
			ev,pT = np.linalg.eig(Kc) 
			fef = np.sqrt(np.abs(2.*ev))	 
			p = pT.transpose() 
			kD = reduce(np.dot, [p, Kc, pT]) 
			kD[kD < 1*10**-13] = 0 
			kDr = np.zeros((n, n)) 
			np.fill_diagonal(kDr, 1/fef)
			#kDr[kDr > 1*10**23] = 0 
			kDrT = kDr.transpose()
			
			######################################
			####            Mass              ####
			######################################
			
			X = b0*(np.random.randn(n, L)) 
			Wc = np.dot(X,(X.T))/L
			mn = reduce(np.dot, [pT,kDrT, Wc, kDr,p]) 
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
			kk = np.exp(k) 
			fef=np.sqrt(kk)
					
			######################################
			####            Mass              ####
			######################################
			
			m = (np.random.uniform(mmin,mmax,(n))) 
			mm = np.exp(m)
			ma_array=np.sqrt(mm)
			mv=1
	
				
			####################################################################
			####################################################################	
			
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