from gen import *
from sim import *
from graph import *
import numpy as np
import os

def experiment():
    T = 1000000
    rho = 0.01
    a1 = np.zeros(T)
    a2 = np.zeros(T)
    a3 = np.zeros(T)
    a4 = np.zeros(T)
    
    for iteration in range(10):
        name = str(iteration)+"-u"
        gen(id = name, seed = (2022+iteration), T = T)
        b1, b2, b3, b4 = sim(id = name, T = T, B = rho*T)
        a1 += b1
        a2 += b2
        a3 += b3
        a4 += b4
        graph(id = name, T = T)
    '''
    for iteration in range(10):
        name = str(iteration)+"-u"
        path = "./output/"+name+"/"
        b1 = np.load(path+"Full_Constrained.npy")
        b2 = np.load(path+"Full_Unconstrained.npy")
        b3 = np.load(path+"Partial_Constrained.npy")
        b4 = np.load(path+"Partial_Unconstrained.npy")
        a1 += b1
        a2 += b2
        a3 += b3
        a4 += b4
    '''
    a1 = 0.1 *a1
    a2 = 0.1 *a2
    a3 = 0.1 *a3
    a4 = 0.1 *a4
    np.save("./output/Uniform/Full_Constrained.npy", a1)
    np.save("./output/Uniform/Full_Unconstrained.npy", a2)
    np.save("./output/Uniform/Partial_Constrained.npy", a3)
    np.save("./output/Uniform/Partial_Unconstrained.npy", a4)

    graph(id = "Uniform", T = T)

if __name__ == "__main__":
    experiment()