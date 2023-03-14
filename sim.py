import numpy as np
from algorithm import *
from config import args

def sim(data_id, alg, B, M, K, epsilon, delta):
    data_path = "./data/"+str(data_id)+"/"
    values = np.load(data_path+"values.npy")
    hobs = np.load(data_path+"hobs.npy")
    T = len(values)

    if alg =='FC':
        average = Full_Constrained(T=T, B=B, M=M, K=K, values = values, hobs = hobs, epsilon=epsilon)
    if alg =='FU':
        average = Full_Unconstrained(T=T, B=B, M=M, K=K, values = values, hobs=hobs)
    if alg =='PC':
        average = Partial_Constrained(T=T, B=B, M=M, K=K, values=values, hobs=hobs, epsilon=epsilon, delta=delta)
    if alg =='PU':
        average = Partial_Unconstrained(T=T, B=B, M=M, K=K, values=values, hobs=hobs, delta=delta)

    plt.plot([idx for idx in range(T)], average)
    plt.show()

    return

if __name__ == "__main__":
    data_id = args.data
    alg = args.alg
    B = args.B
    M = args.M
    K = args.K
    epsilon = args.epsilon
    delta = args.delta
    sim(data_id=data_id, B=B, M=M, K=K, alg = alg, epsilon=epsilon, delta=delta)
