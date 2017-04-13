import numpy as np
from fractions import *
import scipy as sp

###################################################################
####                    Initial rho array                      ####
###################################################################

def rhoinitial(phidotin_array,phiin_array,ma_array,n):
	rhoin_array=[]
	for i in range(n):	
		rhoin_array.append( 0.5*phidotin_array[i]**2 + 0.5*(ma_array[i]**2)*phiin_array[i]**2 )
	#print 'test print rhoinit', rhoin_array
	return rhoin_array
	
###################################################################	

###################################################################
####                    Initial y array                        ####
###################################################################
	
def yinitial(n,phiin_array,phidotin_array,rhoin_array,ain):
	y0=[]
	for i in range(n):
		y0.append(phiin_array[i])
		y0.append(phidotin_array[i])
		y0.append(rhoin_array[i])
	y0.append(ain)
	return y0	
	
###################################################################	

###################################################################
####         Equations of Motion Function (Phi and Rho)        ####
###################################################################

def deriv_wfromphi(y,t,n,n_cross,crossing_index,ma_array,rho_m0,rho_r0,rhol):

	func=[]
	####### Sum rho axions first (sum from y[2] + y[5] + ... + y[3n-1])
	rho_ax=sum(y[2::3])
	Hubble=1./np.sqrt(3.)*np.sqrt(rho_ax + rho_m0/y[-1]**3. + rho_r0/y[-1]**4. + rhol)
	#rho_ax = 0
	#for i in range (n):
		#rho_ax = rho_ax + 0.5*ma_array[i]**2*y[3*i]**2+0.5*y[3*i+1]**2 
	####### Sum rho axions from phi and phidot (Do we need this? which one should we use?)
	#rho_ax_f = 0
	#for i in range(n):
	#	rho_ax_f = rho_ax_f + 0.5*ma_array[i]*y[3*i]*y[3*i] + 0.5*y[3*i+1]*y[3*i+1]
	####### Start filling equations
	####### To check these conditions, we need crossing_index as a global parameter which will be updated every time the condition is met.
	for i in range(n):
		###### update crossing_index
		###### The logic is following:
		###### We start counting when w crosses from negative to positive and crossing index is even
		###### After that crossing index is now odd and we count again when w crosses from positive to negative
		###### After that crossing index is now even and we count again when w crosses from negative to positive
		##### Note that after crossing_index has reached the stated value, it is not meaningful as phi is not accurate any more.
		if ( (crossing_index[i] % 2 == 0 and ma_array[i]**2.*y[3*i]*y[3*i] < y[3*i+1]*y[3*i+1] and y[3*i+1] > 0) or (crossing_index[i] % 2 == 1 and ma_array[i]**2.*y[3*i]*y[3*i] > y[3*i+1]*y[3*i+1] and y[3*i+1] < 0) ):
			crossing_index[i] += 1
			#print crossing_index[i]
		####### phi dot part
		func.append(y[3*i+1])
		#if crossing_index[i] < n_cross:
			#func.append(y[3*i+1])
		#else: ### After cross n_cross times, phidot = 0
			#func.append(0)
		####### phi ddot part
		if crossing_index[i] < n_cross :
			func.append( -3.*Hubble*y[3*i+1] - (ma_array[i]**2)*y[3*i] )
		else: ### After cross n_cross times, the mass term is switched off in order to drag phi down to zero (or simply put phiddot = 0)
			func.append( -3.*Hubble*y[3*i+1] )
			#func.append(0)
		####### rho dot part
		if crossing_index[i] < n_cross  :
			func.append( -3.*Hubble*y[3*i+1]*y[3*i+1] ) ### rho + p = phidot^2
		else: ### After cross n_cross times, P = 0 and rho = 1/a^3
			func.append( -3.*Hubble*y[3*i+2] )	
	# adot
	func.append(Hubble*y[-1])

	return func

