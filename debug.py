import numpy as np

startChain=np.load('Chains/model1_nax20_DE_run2.npy')
pos=startChain[:,-1,:]

print pos