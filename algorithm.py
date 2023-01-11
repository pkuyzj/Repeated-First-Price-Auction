import numpy as np
import math

e = 2.718282

def Full_Constrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    rho = (1.0*B)/T
    K = math.ceil(math.sqrt(T)/50)
    M = K
    #N = np.zeros(M) 
    R = np.zeros([M, K])
    C = np.zeros([K])
    Lambda = 0.0
    Payoff = [0.0]
    # t=0, choose bid b = 0
    for k in range(K):
        if hobs[0]<=k/K:
            C[k] = k/K
            for m in range(M):
                R[m][k] = (m/M)-(k/K)
    # t=1,2,...,T-1   
    for t in range(1,T):
        if t%100==0:
            print("t = "+str(t))
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value/(1+Lambda))*M))
        # Find the best arm in context M
        k_t = 0
        R_max = 0.0
        for k in range(K):
            if R[m_t][k]>R_max: #Note that r = R/n
                R_max = R[m_t][k]
                k_t = k
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff.append(value-b)
        else:
            Payoff.append(0)
        Lambda = min(1/(rho)-1, max(Lambda+epsilon*((C[k_t]/t)-rho),0))
        # Update estimation on all contexts and all arms.
        for k in range(K):
            if hob<=k/K:
                C[k] += k/K
                for m in range(M):
                    R[m][k] += (m/M)-(k/K)
    return Payoff

def Partial_Constrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    rho = (1.0*B)/T
    K = math.ceil(math.sqrt(T)/50)
    M = K
    R = np.zeros([M, K])    # Store the sum. Estimation r = R / n[m][k]
    C = np.zeros([K])       # Store the sum. Estimation c = C / NK[k]
    NK = np.zeros([K])      # Used when computing c bar from C.
    n = np.zeros([M, K])    # n_{m, k}
    Lambda = 0.0
    Payoff = [0.0]
    Active = [{i for i in range(K)} for j in range(M)]
    
    # t=0, choose bid b = 0
    for k in range(K):
        NK[k] = 1.0
        if hobs[0]<=k/K:
            C[k] = k/K    
        for m in range(M):
            n[m][k] += 1.0
            if hobs[0]<k/K:
                R[m][k] += ((m/M)-(k/K))

    for t in range(1,T):
        if t%1000==0:
            print(t)
        # Get value and hob from data
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value/(1+Lambda))*M))
        
        # Bid the smallest bid in context M's active set
        k_t = min(Active[m_t])
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff.append(value-b)
        else:
            Payoff.append(0)
        Lambda = max(Lambda+epsilon*((C[k_t]/NK[k_t])-rho),0)
        
        # Update estimation
        NM = np.zeros(M)+(T+2023) #+\infty
        
        for k in range(k_t, K):
            NK[k] += 1.0
            if k/K>=hob:
                C[k] += (k/K)*1.0
        for m in range(M):
            for k in Active[m]:
                if k>= k_t:
                    n[m][k]+= 1.0
                    if k/K>=hob:
                        R[m][k]+= ((m/M)-(k/K))
                NM[m] = min(NM[m], n[m][k]) 

        # Updating the Active Set for each context.
        bottom = 0  # a variable using to maintain partial-order in elimination.
        CBs = [] #Used in debugging
        for m in range(M): 
            CB =  math.sqrt((2*(math.log(((2*M*K*T)/delta), e)))/(NM[m]))#confidence bound
            CBs.append(CB) #Used in debugging
            to_delete = []
            # 1. Find out the "best" arm.
            r_max = -1.0
            for k in Active[m]:
                if k<bottom:
                    continue
                r_max = max(r_max, R[m][k]/n[m][k])
            # 2. Elimination
            for k in Active[m]:
                if k<bottom:
                    to_delete.append(k)
                    continue
                if R[m][k]/n[m][k] < r_max - 2*CB:
                    to_delete.append(k)
            for k in to_delete:
                Active[m].remove(k)
            bottom = min(Active[m])
        #####The following is used in debugging
        if t%5000 ==0:
            print(CBs[0])
            total_size = 0
            for m in range(M):
                total_size += len(Active[m])
            print("Total Active Set Size: "+str(total_size))
        '''
        if t%500 == 0: 
            for m in range(M):
                if(m%10 ==0):
                    print('m = {0}, N_m = {1}, CB_m = {2}'.format(m, NM[m], CBs[m]))
                    for k in range(K):
                        if(k%10==0):
                            print(format(R[m][k]/n[m][k], '.4f'), end = ',')
                    print("")
        #####'''
    return Payoff

def Partial_Unconstrained(T, B, values, hobs, epsilon, delta):
    Budget = B
    K = math.ceil(math.sqrt(T)/50)
    M = K
    R = np.zeros([M, K])    # Store the sum. Estimation r = R / n[m][k]
    n = np.zeros([M, K])    # n_{m, k}
    Payoff = [0.0]
    Active = [{i for i in range(K)} for j in range(M)]
    
    # t=0, choose bid b = 0
    for k in range(K):  
        for m in range(M):
            n[m][k] += 1.0
            if hobs[0]<k/K:
                R[m][k] += ((m/M)-(k/K))

    for t in range(1,T):
        if t%1000==0:
            print(t)
        # Get value and hob from data
        value, hob = values[t], hobs[t]
        # Locate context m
        m_t = math.floor(((value)*M))
        
        # Bid the smallest bid in context M's active set
        k_t = min(Active[m_t])
        b = min(k_t/K, Budget)
        # Update lambda, budget, payoff
        if hob<= b:
            Budget -= b
            Payoff.append(value-b)
        else:
            Payoff.append(0)
        
        # Update estimation
        NM = np.zeros(M)+(T+2023) #+\infty

        for m in range(M):
            for k in Active[m]:
                if k>= k_t:
                    n[m][k]+= 1.0
                    if k/K>=hob:
                        R[m][k]+= ((m/M)-(k/K))
                NM[m] = min(NM[m], n[m][k]) 

        # Updating the Active Set for each context.
        bottom = 0  # a variable using to maintain partial-order in elimination.
        CBs = [] #Used in debugging
        for m in range(M): 
            CB =  math.sqrt((2*(math.log(((2*M*K*T)/delta), e)))/(NM[m]))#confidence bound
            CBs.append(CB)
            to_delete = []
            # 1. Find out the "best" arm.
            r_max = -1.0
            for k in Active[m]:
                if k<bottom:
                    continue
                r_max = max(r_max, R[m][k]/n[m][k])
            # 2. Elimination
            for k in Active[m]:
                if k<bottom:
                    to_delete.append(k)
                    continue
                if R[m][k]/n[m][k] < r_max - 2*CB:
                    to_delete.append(k)
            for k in to_delete:
                Active[m].remove(k)
            bottom = min(Active[m])
        #####The following is used in debugging
        if t%5000 ==0:
            print(CBs[0])
            total_size = 0
            for m in range(M):
                total_size += len(Active[m])
            print("Total Active Set Size: "+str(total_size))
        '''
        if t%500 == 0: 
            for m in range(M):
                if(m%10 ==0):
                    print('m = {0}, N_m = {1}, CB_m = {2}'.format(m, NM[m], CBs[m]))
                    for k in range(K):
                        if(k%10==0):
                            print(format(R[m][k]/n[m][k], '.4f'), end = ',')
                    print("")
        #####'''
    return Payoff
