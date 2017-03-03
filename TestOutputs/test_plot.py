import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

hz={"out": np.loadtxt('test_out_mod_0.txt',unpack=True)}
for i in range(0,10):
	hz[i]={"out": np.loadtxt('test_out_mod_'+str(i)+'.txt',unpack=True)}

h=0.68

omh2=0.32*h**2.
olh2=0.68*h**2.
orh2=4.16e-05

def hlcdm(z):
	func=np.sqrt(olh2+(1+z)**3.*omh2+(1+z)**4.*orh2)
	return func
	
fig,ax=plt.subplots()

for i in range(0,10):
	z=hz[i]["out"][0]
	H=hz[i]["out"][1]
	plt.plot(z,H)
plt.plot(z,hlcdm(z),'-k',linewidth=line)
plt.plot(z,z/z*h,'--k',alpha=0.5,linewidth=line)

plt.xscale('log')
plt.yscale('log')
#plt.xlim([1.e-2,3403])
#plt.ylim([1.e-1,1.e6])

plt.savefig('test_out_plot.png',bbox_inches = 'tight')


