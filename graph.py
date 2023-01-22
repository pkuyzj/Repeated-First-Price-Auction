import numpy as np
import matplotlib.pyplot as plt

def graph(id, T, pc_only = 0):
    path = "./output/"+str(id)+"/"
    x = [i for i in range(T)]
    
    if pc_only:
      y= np.load(path+"Partial_Constrained.npy")
      y = y.tolist()
      z = y
      s = 0
      for i in range(T):
        s+=y[i]
        y[i] = s
        z[i] = y[i]/(i+1)
      plt.plot(x, z, c = 'r', label = ".")
      f = plt.gcf()
      f.savefig(path+str(id)+"-One-sided, Constraint.png")
      f.clear()
      return
    y1 = np.load(path+"Full_Constrained.npy")
    y1 = y1.tolist()
    z1 = y1
    s = 0
    for i in range(T):
       s+=y1[i]
       y1[i] = s
       z1[i] = y1[i]/(i+1)

    y2 = np.load(path+"Full_Unconstrained.npy")
    y2 = y2.tolist()
    z2 = y2
    s = 0
    for i in range(T):
       s+=y2[i]
       y2[i] = s
       z2[i] = y2[i]/(i+1)

    y3 = np.load(path+"Partial_Constrained.npy")
    y3 = y3.tolist()
    z3 = y3
    s = 0
    for i in range(T):
       s+=y3[i]
       y3[i] = s
       z3[i] = y3[i]/(i+1)
 
    y4 = np.load(path+"Partial_Unconstrained.npy")
    y4 = y4.tolist()
    z4 = y4
    s = 0
    for i in range(T):
       s+=y4[i]
       y4[i] = s 
       z4[i] = y4[i]/(i+1)
    
    plt.figure(1)
    plt.plot(x, z1, c = 'g', label = "Full, Constraint")
    plt.plot(x, z2, c = 'r', label = "Full, No Constraint" )
    f = plt.gcf()
    f.savefig(path+str(id)+"-full.png")
    f.clear()
    '''
    yl = np.load(path+"Lambda.npy")
    yl = yl.tolist()
    plt.figure(1)
    plt.plot(x, yl, c = 'k', label = "Lambda (Onesided, Constraint)")
    f = plt.gcf()
    f.savefig(path+"lambda-"+str(id)+".png")
    f.clear()
    '''
    plt.figure(2)
    plt.plot(x, z3, c = 'b', label = "One-Sided, Constraint")
    plt.plot(x, z4, c = 'r', label = "One-Sided, No constraint")
    f = plt.gcf()
    f.savefig(path+str(id)+"-partial.png")
    f.clear()
if __name__ == "__main__":
    graph("test", 1000000, 1)