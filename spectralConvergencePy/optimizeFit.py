# from http://www.scipy.org/Cookbook/FittingData
from scipy.optimize import leastsq
from numpy import *

class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

def fit(function, parameters,y,x):

    def residuals(parameters, y,x):
        err = y - function(x,parameters)
        return err

    guessfit = function(x,parameters)
    

    best = leastsq(residuals, parameters, args = (y,x), full_output=1)
    #print best
    bestparams = best[0]
    cov_x = best[1]
    
    return bestparams
