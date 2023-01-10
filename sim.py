import numpy as np
import os
import math
from algorithm import *

def sim(id, T, B):
    path = "./output/"+str(id)+"/"
    values = np.load(path+"values.npy")
    assert len(values) == T
    hobs = np.load(path+"hobs.npy")
    assert len(hobs) == T

    epsilon = 1.0/(math.sqrt(T))
    delta = 1.0/T
    
    Payoff1 = Full_Constrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path+"Full_Constrained.npy", Payoff1)
    Payoff2 = Partial_Constrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path + "Partial_Constrained.npy", Payoff2)
    Payoff3 = Partial_Unconstrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path+ "Partial_Unconstrained.npy", Payoff3)
    return 0

if __name__ == "__main__":
    sim(0, 10000, 2000)  #(id, T, B)