import numpy as np
import math

e = 2.718282

def Full_Constrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    rho = (1.0*B)/T
    K = 100
    M = 100
    R = np.zeros([M, K])
    C = np.zeros([K])
    Lambda = 0.0
    v = np.array([1.0*(m/M) for m in range(M)])
    Payoff = np.zeros(T)
    # t=0, choose bid b = 0
    for k in range(K):
        if hobs[0]<=k/K:
            C[k] = k/K
            R[:,k] += (v-(k/K))
    # t=1,2,...,T-1   
    for t in range(1,T):
        if t%50000==0:
            print(str(t)+" / "+ str(T))
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value/(1+Lambda))*M))
        # Find the best arm in context M
        k_t = R[m_t].argmax()
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff[t] = (value-b)

        Lambda =max(Lambda+(1.0/(math.sqrt(T)))*((C[k_t]/t)-rho),0)
        # Update estimation on all contexts and all arms.
        for k in range(K):
            if hob<=k/K:
                C[k] += k/K
                R[:,k] += (v-(k/K))
    return Payoff

def Full_Unconstrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    rho = (1.0*B)/T
    K = 100
    M = 100
    R = np.zeros([M, K])
    v = np.array([1.0*(m/M) for m in range(M)])
    Payoff = np.zeros(T)
    # t=0, choose bid b = 0
    for k in range(K):
        if hobs[0]<=k/K:
            R[:,k] += (v-(k/K))
    # t=1,2,...,T-1   
    for t in range(1,T):
        if t%50000==0:
            print(str(t)+" / "+ str(T))
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value)*M))
        # Find the best arm in context M
        k_t = R[m_t].argmax()
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff[t] = (value-b)
        # Update estimation on all contexts and all arms.
        for k in range(K):
            if hob<=k/K:
                R[:,k] += (v-(k/K))
    return Payoff

def Partial_Constrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    rho = (1.0*B)/T
    M = 100
    K = 100
    R = np.zeros([M, K])    # Store the sum. Estimation r = R / n[m][k]
    C = np.zeros(K)       # Store the sum. Estimation c = C / NK[k]
    n = np.zeros(K)
    v = np.array([1.0*(m/M) for m in range(M)])
    r = np.zeros([M, K])
    Lambda = 0.0
    Payoff = np.zeros(T)
    Active = np.ones([K,M]) #[np.array([k for k in range(K)]) for m in range(M)]
    l = np.zeros(M, dtype=np.int32)
    Lambdas = np.zeros(T)
    # t=0, choose bid b = 0
    for k in range(K):
        n[k] = 1.0
        if hobs[0]<=k/K:
            C[k] = k / K
            R[:,k] += (v-(k/K))

    for t in range(1,T):
        if t%50000==0:
            print(str(t)+" / "+ str(T))
        # Get value and hob from data
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value/(1+Lambda))*M))
        # Bid the smallest bid in context M's active set
        k_t = int(l[m_t])
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff[t] = (value-b)
        #Lambda = Lambda * math.pow(math.e, ((C[k_t]/n[k_t])-rho)/(math.sqrt(T)))
        Lambda = max(Lambda+(1.0/(math.sqrt(T)))*((C[k_t]/n[k_t])-rho),0)
        Lambdas[t] = Lambda
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
        r = r*Active #################
        # Updating the Active Set for each context.
        #CB1 = 0.0
        #CB2 = 0.0
        for m in range(M): 
            #CB =  math.sqrt(((6+3*math.log(2*t-1))*math.log(M*K*T/delta))/(n[l[m]]))
            #CB =  math.sqrt(((6+3*math.log(2*t-1))*math.log(M*K*T/delta))/(n[l[m]]))
            CB = (m/M) * math.sqrt(((6+3*math.log(2*t-1))*math.log(K*T/delta))/(n[l[m]]))
            # 1. Find out the "best" arm.
            r_max = r[m].max()
            r[m] = r[m] - r_max + 2*CB
            #if m==0:
                #CB1 = CB
            #CB2 = CB
        
        # 2. Elimination
        Active = Active*(r>0)
        
        for m in range(M):
            for k in range(l[m], K):
                if Active[m][k]!=0:
                    l[m] = k
                    break
        #####The following is used in debugging
        if t%50000 ==0:
            total_size = 0
            #print(CB1)
            #print(CB2)
            for m in range(M):
                total_size += sum(Active[m])
            print("Total Active Set Size: "+str(total_size))
    return Payoff, Lambdas

def Partial_Unconstrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    M = 100
    K = 100
    R = np.zeros([M, K])    # Store the sum. Estimation r = R / n[m][k]
    n = np.zeros(K)
    v = np.array([1.0*(m/M) for m in range(M)])
    r = np.zeros([M, K])
    Payoff = np.zeros(T)
    Active = np.ones([K,M]) #[np.array([k for k in range(K)]) for m in range(M)]
    l = np.zeros(M, dtype=np.int32)
    # t=0, choose bid b = 0
    for k in range(K):
        n[k] = 1.0
        if hobs[0]<=k/K:
            R[:,k] += (v-(k/K))

    for t in range(1,T):
        if t%50000==0:
            print(str(t)+" / "+ str(T))
        # Get value and hob from data
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value)*M))
        # Bid the smallest bid in context M's active set
        k_t = int(l[m_t])
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff[t] = (value-b)
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
        r = r*Active #################
        # Updating the Active Set for each context.
        for m in range(M):  
            #CB =  math.sqrt((6*(math.log(2*t))*(math.log(((2*M*K*T)/delta), math.e)))/(n[l[m]]))#confidence bound
            CB = (m/M) * math.sqrt(((6+3*math.log(2*t-1))*math.log(K*T/delta))/(n[l[m]]))
            # 1. Find out the "best" arm.
            r_max = r[m].max()
            r[m] = r[m] - r_max + 2*CB
            
        # 2. Elimination
        Active = Active*(r>0)
        
        for m in range(M):
            for k in range(l[m], K):
                if Active[m][k]!=0:
                    l[m] = k
                    break
        #####The following is used in debugging
        '''
        if t%50000 ==0:
            total_size = 0
            for m in range(M):
                total_size += sum(Active[m])
            print("Total Active Set Size: "+str(total_size))
        '''
    return Payoff