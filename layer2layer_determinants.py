import matplotlib.pyplot as plt
import numpy as np


softplus=lambda x:np.log(1+np.exp(x))
lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
relu=lambda x:np.maximum(0.,x)
exp=np.exp
exp_minus1=lambda x: np.exp(x)-1
#elu=lambda x: np.max(0.,x)+np.int(x<0)*(0.2)*(np.exp(x)-1.)
elu=lambda x: np.maximum(np.exp(x)-1.,x)####Warn this was wrong!!!!
#elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
tanh=np.tanh


def sorted_random_positive_numbers(a=0,b=5,N=5):
    weights=np.random.uniform(a,b,N)
    weights.sort()
    return weights


#Vandermonde basis
k=5
kprev=7#num basis funcs in previous layer
vand_weights=sorted_random_positive_numbers(N=kprev)
vand_locations=sorted_random_positive_numbers(N=k)

#Should have pos determinant
Vprev=vand_locations.reshape([1,-1])**(vand_weights.reshape([-1,1]))#kprev x k


#positive cone-ordered weights
weights=np.random.rand(k,kprev).cumsum(axis=0)

#New Vand matrix






