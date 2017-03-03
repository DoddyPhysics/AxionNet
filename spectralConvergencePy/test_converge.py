import spectralConvergence as sd

root = '/Users/davidmarsh/Physics/MultiAxion/BayesNetAxion/Chains/'
filename = 'model1_nax20_DE'
iterations=3
burnfrac=5 

sd.converge(root,filename+'.npy',numiter=iterations,burnin=burnfrac)
