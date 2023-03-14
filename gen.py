import numpy as np
import os
import math
from config import args
def gen(data_id, seed, T, distribution, param_1, param_2, mu, sigma):
    path = os.getcwd()+"/data/"+str(data_id)+"/"
    if os.path.exists(path) == False:
        os.makedirs(path)

    np.random.seed(seed)

    if distribution == 'n':
        #NORMAL DISTRIBUTION
        values = []
        while len(values) < T:
            value = np.random.normal(param_1, param_2)
            if value>=0.0 and value <= 1.0:
                values.append(value)
        values = np.array(values)
    
    if distribution == 'u':
        #UNIFORM DISTRIBUTION
        values = np.random.uniform(param_1, param_2, T)

    if distribution == 'l':
        #LOGARITHMIC NORMAL DISTRIBUTION
        values = []
        while len(values) < T:
            logv = np.random.normal(param_1, param_2)
            value = math.pow(math.e, logv)
            if value>=0.0 and value <= 1.0:
                values.append(value)
        values = np.array(values)

    hobs = []
    while len(hobs) < T:
        hob = np.random.normal(mu, sigma)
        if hob>=0.0 and hob <= 1.0:
            hobs.append(hob)   
    hobs = np.array(hobs)

    np.save(path+"values.npy", values)
    np.save(path + "hobs.npy", hobs)
    
if __name__ == "__main__":
    print(args)
    data_id = args.data
    seed = args.seed
    T = args.T
    distribution = args.distribution
    param1 = args.param1
    param2 = args.param2
    mu = args.mu
    sigma = args.sigma
    gen(data_id = data_id, seed = seed, T = T, distribution=distribution, param_1=param1, param_2=param2, mu=mu, sigma=sigma) 
