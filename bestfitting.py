import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import minimize

global par

##Download the data
data_file = "notrypsin.dat" 
trypsinfile = np.loadtxt(data_file)
tdata=trypsinfile[:,0]
vdata=trypsinfile[:,1]

#=====================================================
#Notice we must import the Model Definition
##System of ODEs
def gammam_dt(Z,t):
    global par
    b = par[0]
    prod = par[1]
    clear = par[2]
    nI = 100
    nE =  int(np.ceil(par[3]))
    d = nI / par[5]
    k = nE / par[4]
    N = 1
#   T, E, I, D, N, V = 0, 1, 2, np.arange(3, 3 + nE), np.arange(3 + nE, 3 + nE + nI), np.sum(np.arange(T, D + 1))
    gammam = []
    gammam.append(-(b / N) * Z[1+nE+nI] * Z[0])
    gammam.append((b / N) * Z[1+nE+nI] * Z[0] - (k * Z[1]))
    for i in range(1,nE):
        gammam.append(k * (Z[i]-Z[i+1]))  
    gammam.append(k * Z[nE] - (d * Z[1+nE]))   
    for i in range(1, nI):
        gammam.append(d * (Z[nE+i]-Z[nE+i+1]))   
    gammam.append(prod * np.sum(Z[1+nE:nE+nI]) - (clear * Z[1+nE+nI]))
    return gammam

#=====================================================
# model index to compare to data
#inside of ssr
#3.Score Fit of System
#=========================================================
def SSR(pp):
    global par
    global trypsinfile
    global ndata

    fpar=10**(pp)
    timecolumn = trypsinfile[:,0]
    par = []
    par.extend(fpar[:])
    nI = int(100)
    nE = int(np.ceil(fpar[3]))
    dset = [0] 
    dset.extend(np.where(np.diff(timecolumn) <= 0)[0]+1) 
    dset.append(len(trypsinfile))
    upd = []
    for i in np.arange(len(dset)-1):
        sample_time = [0];
        sample_time.extend(timecolumn[dset[i]:dset[i+1]])
        Z = [1]
        Z.extend(np.zeros(nE+nI))
        Z.append(fpar[-1]*10**(i-1))
        y = odeint(gammam_dt, Z, sample_time)
        for l in np.where(y[1:,-1]<1.0)[0]+1:
            y[l,-1]=1.0
        upd.extend(y[1:,-1])
        plt.plot(np.log10(vdata),'ro')
        plt.plot(np.log10(upd))
        plt.show()
        print(par)
		#upd virus from parameters
    def ss(dat, mod):
        summ=0
        for i in range(len(dat)): 
            if dat[i]>=0:
            	summ=summ+(dat[i]-mod[i])**2
        return summ	
    return ss(np.log10(vdata), np.log10(upd))
#========================================================
 

#2.Set up Info for Model System
#===================================================
# model parameters
# model initial conditions
#---------------------------------------------------
bfpar=[  8.29751649e-05,   8.13646143e+03,   7.26734988e+00,  
 1.95704245e+00,  .500000000e+02,   5.50343473e+01,   6.87212224e+02]

#4.Optimize Fit
#=======================================================
#fit_score=score(rates)
fpar = np.log10(bfpar)
bnds=((None,None),(None,None),(None,None),(0,2),(0,2),(0,2),(None,None))
answ = minimize(SSR,fpar, method='L-BFGS-B', bounds=bnds)
#answ = minimize(SSR,fpar,method='Nelder-Mead')
bestrates=answ.x
bestscore=answ.fun
print(10**bestrates,bestscore)
#=======================================================
#