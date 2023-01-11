import numpy as np
import os

def gen(id, seed, T):
    path = os.getcwd()+"/output/"+str(id)+"/"
    if os.path.exists(path) == False:
        os.makedirs(path)

    #torch.manual_seed(seed)
    np.random.seed(seed)
    values = []
    hobs = []

    while len(values) < T:
        value = np.random.normal(0.5, 0.1)
        if value>=0.0 and value <= 1.0:
            values.append(value)
    while len(hobs) < T:
        hob = np.random.normal(0.4, 0.1)
        if hob>=0.0 and hob <= 1.0:
            hobs.append(hob)    

    values = np.array(values)
    hobs = np.array(hobs)

    path_values = path+"values.npy"
    path_hobs = path + "hobs.npy"

    np.save(path_values, values)
    np.save(path_hobs, hobs)

if __name__ == "__main__":
    gen(0, 2023, 1000000)
