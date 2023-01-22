import numpy as np
import os
import math
def gen(id, seed, T):
    path = os.getcwd()+"/output/"+str(id)+"/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    #torch.manual_seed(seed)
    np.random.seed(seed)

    #NORMAL DISTRIBUTION
    '''
    values = []
    while len(values) < T:
        value = np.random.normal(0.5, 0.1)
        if value>=0.0 and value <= 1.0:
            values.append(value)
    values = np.array(values)
    '''
    
    #UNIFORM DISTRIBUTION
    values = np.random.uniform(0.25, 1.0, T)
            #hobs = np.random.uniform(0.2, 0.8, T)

    '''
    #LOGARITHMIC NORMAL
    
    values = []
    while len(values) < T:
        logv = np.random.normal(-0.5, 0.1)
        value = math.pow(math.e, logv)
        if value>=0.0 and value <= 1.0:
            values.append(value)
    values = np.array(values)
    '''

    hobs = []
    while len(hobs) < T:
        hob = np.random.normal(0.4, 0.1)
        if hob>=0.0 and hob <= 1.0:
            hobs.append(hob)   
    hobs = np.array(hobs)

    path_values = path+"values.npy"
    path_hobs = path + "hobs.npy"
    np.save(path_values, values)
    np.save(path_hobs, hobs)
    
if __name__ == "__main__":
    gen("test", 2023, 1000000) #Seed = 2022+id