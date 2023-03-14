import numpy as np
import math
from tqdm import trange
import matplotlib.pyplot as plt

def Full_Constrained(T, B, M, K, values, hobs, epsilon):
    Budget = B
    rho = (1.0*B)/T
    R = np.zeros([M, K])
    C = np.zeros([K])
    Lambda = 0.0
    v = np.array([1.0*(m/M) for m in range(M)])
    Payoff = 0.0
    average = np.zeros(T)
    
    bar = trange(T)
    for t in bar:
        bar.set_description("Total utility:{:.2f}, Remaining budget:{:.2f}".format(Payoff,Budget))
        value, hob = values[t], hobs[t]
        if t==0:
            for k in range(K):
                if hob<=k/K:
                    C[k] = k / K
                    R[:,k] += (v-(k/K))
            continue
        
        # Locate context m
        m_t = math.floor(((value/(1+Lambda))*M))
        
        # Find the best arm in context M
        k_t = R[m_t].argmax()
        b = min(k_t/K, Budget)
        
        # Update lambda, budget, payoff
        if hob<= b:            
            Budget -= b
            Payoff += (value-b)

        Lambda =max(Lambda+epsilon*((C[k_t]/t)-rho),0)

        # Update estimation on all contexts and all bids.
        for k in range(K):
            if hob<=k/K:
                C[k] += k/K
                R[:,k] += (v-(k/K))

        average[t] = Payoff/(1.0*t)

    return average


def Full_Unconstrained(T, B, M, K, values, hobs):
    Budget = B
    R = np.zeros([M, K])
    v = np.array([1.0*(m/M) for m in range(M)])
    Payoff = 0.0
    average = np.zeros(T)
    
    bar = trange(T)
    for t in bar:
        bar.set_description("Total utility:{:.2f}, Remaining budget:{:.2f}".format(Payoff,Budget))
        value, hob = values[t], hobs[t]
        if t==0:
            for k in range(K):
                if hob<=k/K:
                    R[:,k] += (v-(k/K))
            continue

        if Budget<= 0:
            average[t] = Payoff / (1.0*t)
            continue

        # Locate context m
        m_t = math.floor(((value)*M))
        
        # Find the best bid in context M
        k_t = R[m_t].argmax()
        b = min(k_t/K, Budget)
        
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff += (value-b)
        
        # Update estimation on all contexts and all bids.
        for k in range(K):
            if hob<=k/K:
                R[:,k] += (v-(k/K))
        average[t] = Payoff / (1.0*t)

    return average

def Partial_Constrained(T, B, M, K, values, hobs, epsilon, delta):
    Budget = B
    rho = (1.0*B)/T
    R = np.zeros([M, K])    # Store the sum. Estimation r = R / n[m][k]
    C = np.zeros(K)       # Store the sum. Estimation c = C / NK[k]
    n = np.zeros(K)
    v = np.array([1.0*(m/M) for m in range(M)])
    r = np.zeros([M, K])
    Lambda = 0.0
    average = np.zeros(T)
    Payoff = 0.0
    Active = np.ones([K,M])
    l = np.zeros(M, dtype=np.int32)

    bar = trange(T)
    for t in bar:
        bar.set_description("Total utility:{:.2f}, Remaining budget:{:.2f}".format(Payoff,Budget))
        value, hob = values[t], hobs[t]

        if t == 0:
            for k in range(K):
                n[k] = 1.0
                if hob<=k/K:
                    C[k] = k / K
                    R[:,k] += (v-(k/K))
            continue

        # Locate context m
        m_t = math.floor(((value/(1+Lambda))*M))

        # Bid the smallest bid in the active set of m_t
        k_t = int(l[m_t])
        b = min(k_t/K, Budget)

        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff += (value-b)

        Lambda = max(Lambda+epsilon*((C[k_t]/n[k_t])-rho),0)

        # Elimination: Partial Order
        for m in range(1, M):
            l[m] = max(l[m], l[m-1])
            Active[m][:(l[m])] = 0.0

        # Update estimation
        for k in range(k_t, K):
            n[k] += 1.0
            if k/K>=hob:
                C[k] += (k/K)*1.0
                R[:,k] += (v-(k/K))
            r[:,k] = R[:,k]/n[k]
            
        r = r*Active 
        
        # Updating the Active Set for each context.
        for m in range(M): 
            CB = (m/M) * math.sqrt(((6+3*math.log(2*t-1))*math.log(K*T/delta))/(n[l[m]]))
            r_max = r[m].max()
            r[m] = r[m] - r_max + 2*CB
        
        #Elimination: Confidence Bound
        Active = Active*(r>0)
        
        for m in range(M):
            for k in range(l[m], K):
                if Active[m][k]!=0:
                    l[m] = k
                    break
        
        average[t] = Payoff / (1.0*t)

    return average

def Partial_Unconstrained(T, B, M, K, values, hobs, delta):
    Budget = B
    R = np.zeros([M, K])    # Store the sum. Estimation r = R / n[m][k]
    n = np.zeros(K)
    v = np.array([1.0*(m/M) for m in range(M)])
    r = np.zeros([M, K])
    average = np.zeros(T)
    Payoff = 0.0
    Active = np.ones([K,M]) 
    l = np.zeros(M, dtype=np.int32)

    bar = trange(T)
    for t in bar:
        bar.set_description("Total utility:{:.2f}, Remaining budget:{:.2f}".format(Payoff,Budget))
        value, hob = values[t], hobs[t]
        if t==0:
            for k in range(K):
                n[k] = 1.0
                if hob<=k/K:
                    R[:,k] += (v-(k/K))
            continue

        if Budget<=0:
            average[t] = Payoff/(1.0*t)
            continue

        # Locate context m
        m_t = math.floor(((value)*M))

        # Bid the smallest bid in the active set of M_t
        k_t = int(l[m_t])
        b = min(k_t/K, Budget)

        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff += (value-b)

        # Elimination: Partial Order
        for m in range(1, M):
            l[m] = max(l[m], l[m-1])
            Active[m][:(l[m])] = 0.0

        # Update estimation
        for k in range(k_t, K):
            n[k] += 1.0
            if k/K>=hob:
                R[:,k] += (v-(k/K))
            r[:,k] = R[:,k]/n[k]

        r = r*Active

        # Updating the Active Set for each context.
        for m in range(M):  
            CB = (m/M) * math.sqrt(((6+3*math.log(2*t-1))*math.log(K*T/delta))/(n[l[m]]))
            r_max = r[m].max()
            r[m] = r[m] - r_max + 2*CB
            
        Active = Active*(r>0)
        
        for m in range(M):
            for k in range(l[m], K):
                if Active[m][k]!=0:
                    l[m] = k
                    break

        average[t] = Payoff / (1.0*t)
        
    return average
