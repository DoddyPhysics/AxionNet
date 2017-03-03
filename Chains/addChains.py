import numpy as np

# Axes of chains are nwalker,nsteps,nparams

chain1=np.load('model1_nax20_DE.npy')
chain2=np.load('model1_nax20_DE_run3.npy')

newchain=np.concatenate((chain1,chain2),axis=1)

np.save('model1_nax20_DE.npy',newchain)