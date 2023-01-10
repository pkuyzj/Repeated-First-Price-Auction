import numpy as np
import matplotlib.pyplot as plt

def graph(id, T):
    path = "./output/"+str(id)+"/"
    x = [i for i in range(T)]
    y1 = np.load(path+"Full_Constrained.npy")
    y1 = y1.tolist()
    s = 0
    for i in range(T):
       s+=y1[i]
       y1[i] = s 
    y2 = np.load(path+"Partial_Constrained.npy")
    y2 = y2.tolist()
    s = 0
    for i in range(T):
       s+=y2[i]
       y2[i] = s 
    y3 = np.load(path+"Partial_Unconstrained.npy")
    y3 = y3.tolist()
    s = 0
    for i in range(T):
       s+=y3[i]
       y3[i] = s 
    plt.plot(x, y1, c = 'g', label = "Full, Constraint")
    plt.plot(x, y2, c = 'b', label = "One-Sided, Constraint")
    plt.plot(x, y3, c = 'r', label = "One-Sided, No constraint")
    plt.show()
if __name__ == "__main__":
    graph(0, 10000)