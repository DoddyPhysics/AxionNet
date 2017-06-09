import corner as triangle
import numpy as np
from matplotlib import rcParams

run_name='Mtheory_nax20_DM_run1'
chain=np.load(run_name+'.npy')
nwalkers, nsteps,ndim = np.shape(chain)
burnin = nsteps/4
# Make sample chain removing burnin 
combinedUSE=chain[:,burnin:,:].reshape((-1,ndim))

# Priors
lFL3min,lFL3max=100.,115.
sminl,sminu=10.,30.
smaxl,smaxu=30.,100.
Nmin,Nmax=0.5,1.
betamin,betamax=0.,1.


#################################
# Plotting fonts
###############################

F1 = 20 # Axes font size
F2 = 20 # Legend font size
line = 1.5 # Line width

# Setting the font structure

rc = rcParams # Font structure is called rc now
rc['text.usetex'] = True # Tex fonts
rc['font.family'] = 'serif'
rc['font.serif'].insert(0,'cm') # Default font is computer modern for latex
rc['font.size'] = F1
rc['xtick.labelsize'] = 'small'
rc['ytick.labelsize'] = 'small'
rc['legend.fontsize'] = F2


##############################
# Binning
#############################


bins=20
# Linear binning for linear prior
lFbins=np.linspace(lFL3min,lFL3max,num=bins)
sminbins=np.linspace(sminl,sminu,num=bins)
smaxbins=np.linspace(smaxl,smaxu,num=bins)
Nbins=np.linspace(Nmin,Nmax,num=bins)
betabins=np.linspace(betamin,betamax,num=bins)

#############################################
# Triangle plot: show 1 and 2 sigma levels following triangle documentation
###########################################

combinedCOL='#7E1946'


fig2 = triangle.corner(combinedUSE, labels=[r'$\log_{10}F\Lambda^3$',r'$s_{\rm min}$',r'$s_{\rm max}$',r'$\widetilde{N}$',r'$\beta_\mathcal{M}$'],
	color=combinedCOL,smooth1d=2,smooth=2.,plot_datapoints=False,
	levels=(1-np.exp(-0.5),1-np.exp(-2.)),
	density=True,range=[[lFL3min,lFL3max],[sminl,sminu],[smaxl,smaxu],[Nmin,Nmax],[betamin,betamax]],
	bins=[lFbins,sminbins,smaxbins,Nbins,betabins])
	


fig2.savefig('Plots/'+run_name+"_triangle.pdf")
fig2.savefig('Plots/'+run_name+"_triangle.png")