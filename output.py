import numpy as np

###################################################################
###################################################################
####                        Output                             ####
###################################################################
###################################################################

#########################################
#####     Axion Energy Density     ######
#########################################
def axionrho(y,N,n):
    rhoa = np.sum(y[:,2::3][:],axis=-1)
    return rhoa
#########################################
	
#########################################
####	   	Phi		     ####
#########################################

def axionphi(y,N):
	phi = np.sum(y[:,0:-1:3][:],axis=-1)
	return phi 
def axionphidot(y,N):
	phid = np.sum(y[:,1::3][:],axis=1)
	return phid

#########################################
	
#########################################
##  Matter & Radiation energy density  ##
#########################################
def dense(rho_m0,rho_r0,N,y):
	rhom=[]
	rhor=[]
	for i in range(N):
		rhom.append(rho_m0/y[:,-1][i]**3.)
		rhor.append(rho_r0/y[:,-1][i]**4.)
	return rhom,rhor
#########################################
	
#########################################
###	Cosmological Constant	      ###
#########################################

def clambda(rhol,N):
	rholl=[]
	for i in range (N):
		rholl.append(rhol)
	return rholl

#########################################

#########################################
#####    	 Pressure          ######
#########################################	
def pressure(y,ma_array,N,n,rhom,rhol,rhor):

	Parray = np.zeros((N,n))
	##########################
	# for debugging, make a density array in here and plot the equation of state to make sure it is sensible
	#densarray =np.zeros((N,n))
	##############################
# We must be careful to calculate the pressure accroding to the crossing index condition.
# The axion field i is y[:,3i]. With the eom mass turned off after n_cross, the field never oscillates after this.
# Locate the last zero crossing of the field (http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python).
# Only fill the Parray with non-zero values up to the final zero crossing index.
	
	for i in range(n):
		field=y[:,3*i]
		zero_crossings = np.where(np.diff(np.sign(field)))[0]
		if np.size(zero_crossings)==0:
			last_zero=N
		else:
			last_zero=zero_crossings[-1]
		for j in range(last_zero):
			Parray[j,i]=0.5*y[j,3*i+1]**2.-0.5*ma_array[i]**2.*field[j]**2.
		#	densarray[j,i]=y[j,3*i+2]
		#for j in range(last_zero,N):
		#	densarray[j,i]=y[j,3*i+2]			

# Test w plot for debugging			
		#import matplotlib.pyplot as plt
		#wtest=Parray[:,i]/densarray[:,i]
		#plt.plot(wtest)
	#plt.show()

	P=np.sum(Parray,axis=1) # sum up for all axions

	phom = np.array(rhom)*0.
	phol = np.array(rhol)*1./3.
	phor = np.array(rhor)*-1.
	
	Psum = phol+phor+phom+P


	return P,Psum
#########################################	

#########################################
########   Total energy density  ########
#########################################
def totalrho(rhom,rhol,rhor,rhoa,N):
    rhom = np.array(rhom)
    rhol = np.array(rhol)
    rhor = np.array(rhor)
    rhoa = np.array(rhoa)
    rhosum = rhom+rhol+rhor+rhoa
    return rhosum
#########################################

#########################################
#########        W Axion         ########
#########################################
def w(P,rhoa,N):
	w = P/rhoa
	return w
#########################################

#########################################
########   	Omegas  	########
#########################################
def omegas(rhom,rhor,rhosum,N):
	omegar=[]
	omegam=[]
	for i in range(N):
    		omegar.append(rhom[i]/rhosum[i])
    		omegam.append(rhor[i]/rhosum[i])
    	return omegam,omegar
#########################################

#########################################
########       Hubble scale      ########
#########################################
def hubble(t,rhosum):	
    H = (1.0/np.sqrt(3.0))*np.sqrt(rhosum[0:len(t)])
    return H
#########################################
	
#########################################
########       Scale Factor      ########
#########################################

def scalefactor(y,N):
	a=y[:,-1][0:N]
	return a

#########################################

#########################################
######## Acceloration Equation  ########
#########################################

def accel(a,N,rhosum,psum):
	add=[]
	for ii in range(N):
		add.append(-a[ii]/3*(rhosum[ii]+3*psum[ii]))
	return add

#########################################

#########################################
#########        Redshift       ########
#########################################
def redshift(y,N):
    z=[]
    z = 1.0/y[:,-1][0:N] - 1
	#for i in range(N):
	#	z.append(1/y[:,-1][i] - 1)	
    return z	
#########################################

###################################################################
####             Dark Matter and Dark Energy Densities of axions      ####
###################################################################	
def darkflow(y,N,n):
	# Initialise arrays at zero
	rhoDMarray = np.zeros((N,n))
	rhoDEarray = np.zeros((N,n))
	
	for i in range(n):
		field=y[:,3*i]
		zero_crossings = np.where(np.diff(np.sign(field)))[0]
		if np.size(zero_crossings)==0:
			last_zero=N
		else:
			last_zero=zero_crossings[-1]
		for j in range(0,last_zero):
			# DE density is filled up to last_zero
			rhoDEarray[j,i]=y[j,3*i+2]
		for j in range(last_zero,N):
			# DM density is filled after last zero
			rhoDMarray[j,i]=y[j,3*i+2]

	rhoDM=np.sum(rhoDMarray,axis=1) # sum up for all axions
	rhoDE=np.sum(rhoDEarray,axis=1) # sum up for all axions
	
	return rhoDM,rhoDE

def zeq(z,rhoM,rhoR):
	"""
	Compute redshift of matter radiation equality.
	Should raise an exception in case this is not found.
	"""
	condition=rhoR-rhoM
	inds=np.where(condition<0)[0]
	equality=z[inds[0]]
	
	return equality	

###################################################################
###################################################################
				
