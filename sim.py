import numpy as np
import os
import math
from algorithm import *

def sim(id, T, B, l3 = 0):
    path = "./output/"+str(id)+"/"
    values = np.load(path+"values.npy")
    assert len(values) == T
    hobs = np.load(path+"hobs.npy")
    assert len(hobs) == T

    epsilon = 1.0/(math.sqrt(T))
    delta = 0.01
    
    if l3 == 1:
        Payoff3, Lambdas = Partial_Constrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
        np.save(path + "Partial_Constrained.npy", Payoff3)
        np.save(path+"Lambda.npy", Lambdas)
        return 0

    Payoff1 = Full_Constrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path+"Full_Constrained.npy", Payoff1)
    Payoff2 = Full_Unconstrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path+"Full_Unconstrained.npy", Payoff2)
    Payoff3, Lambdas = Partial_Constrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path + "Partial_Constrained.npy", Payoff3)
    np.save(path+"Lambda.npy", Lambdas)
    Payoff4 = Partial_Unconstrained(T=T, B=B, values=values, hobs=hobs, epsilon = epsilon, delta = delta)
    np.save(path+ "Partial_Unconstrained.npy", Payoff4)
    
    return Payoff1, Payoff2, Payoff3, Payoff4

if __name__ == "__main__":
    sim("test", 1000000, 10000, 1)