import corner as triangle
import numpy as np
from matplotlib import rcParams

run_name='model1_nax20_DM'
chain=np.load(run_name+'.npy')
nwalkers, nsteps,ndim = np.shape(chain)
burnin = nsteps/4
# Make sample chain removing burnin 
combinedUSE=chain[:,burnin:,:].reshape((-1,ndim))

# Priors, for plotting limits and binning
fmin,fmax=-1.8,-1.
betamin,betamax=0.,1.
b0min,b0max=5.8,8.

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
fbins=np.linspace(fmin,fmax,num=bins)
b0bins=np.linspace(b0min,b0max,num=bins)
betabins=np.linspace(betamin,betamax,num=bins)

#############################################
# Triangle plot: show 1 and 2 sigma levels following triangle documentation
###########################################

combinedCOL='#2F52E0'


fig2 = triangle.corner(combinedUSE, labels=[r'$\log_{10}\bar{f}$', r'$\beta_{\mathcal{M}}$',r'$\log_{10}\sigma_{\mathcal{M}}$'],
	color=combinedCOL,smooth1d=2,smooth=2.,plot_datapoints=False,
	levels=(1-np.exp(-0.5),1-np.exp(-2.)),
	density=True,range=[[fmin,fmax],[betamin,betamax],[b0min,b0max]],bins=[fbins,betabins,b0bins])
	


fig2.savefig('Plots/'+run_name+"_triangle.pdf")
fig2.savefig('Plots/'+run_name+"_triangle.png")