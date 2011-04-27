from math import *
from numeric import *
from copy import copy, deepcopy
import time
import random

__all__ = ['AR1filter', 'test_AR1filter',
           'truncate_eigenvalues', 'cov2cor', 'cor2cov', 'cor2cor',
           'truncate_eigenvalues_cor', 'truncate_eigenvalues_cov', 
           'test_truncate_eigenvalues_cov', 'mean_LA', 'covariance_LA', 'test_all']


def AR1filter(r):
    r.sort()    
    """Performs AR(1) filtering and eliminates auto-correlation"""
    x=[]
    y=[]
    for i in range(1,len(r)):
        x.append(r[i-1][1])
        y.append(r[i][1])
        pass
    try:
        a,chi2,ff=fit(LINEAR,x,y)
        a1=a[1][0]
    except:
        a=[]
        a1=0
    r2=[r[0]]
    for i in range(1,len(r)):
        r2.append((r[i][0],(r[i][1]-a1*r[i-1][1])/(1.0-a1)))
        pass
    r2.reverse()
    return r2,a

def test_AR1filter():
    """Test for AR1filter"""
    print "\n\nTesting AR1filter()........."
    y=[(0,0.05)]
    for k in range(1,30):
        y.append((k,0.05+y[k-1][1]*0.3+random.gauss(0,0.01)))
        pass
    yf,a=AR1filter(y)
    for k in range(30):
        print y[k],',',yf[k]
        pass
    return

def truncate_eigenvalues(A,delta=0.01,checkpoint=None):
    """Takes a symmetric matrix and relaces all eigenvalues with < delta with delta"""
    U,e1=Jacobi(A,checkpoint)
    e2=deepcopy(e1)
    for i in range(len(e2)):
        if e2[i]<delta: e2[i]=delta
        pass
    return multiply(U,multiply(diagonal(e2),transpose(U))),e1,e2

def cov2cor(cov):
    n=rows(cov)
    sigma=[0]*n
    cor=matrix(n,n)
    for i in range(n):
        sigma[i]=sqrt(cov[i][i])
        for j in range(0,i+1):
            cor[i][j]=cor[j][i]=cov[i][j]/sigma[i]/sigma[j]            
            pass
        pass
    return cor, sigma

def cor2cov(cor,sigma):
    n=rows(cor)
    cov=matrix(n,n)    
    for i in range(n):
        for j in range(0,i+1):
            cov[i][j]=cov[j][i]=cor[i][j]*sigma[i]*sigma[j]
            pass
        pass
    return cov

def cor2cor(cor):
    """deprecated!"""
    n=rows(cor)
    for i in range(n):
        for j in range(0,i):
            a=cor[i][j]
            b=abs(a)
            if b>1: cor[i][j]=cor[j][i]=a/b
            pass
        pass
    return cor

def truncate_eigenvalues_cor(cor,delta=0.01,checkpoint=None):
    """like truncate_eigenvalues but restores the diagonal elements to 1"""
    if len(cor)<2: return cor,[1],[1]
    cov2,e1,e2=truncate_eigenvalues(cor,delta,checkpoint)
    cor,sigma2=cov2cor(cov2) # restore 1 on diagonal
    return cor,e1,e2

def truncate_eigenvalues_cov(cov,delta=0.01,checkpoint=None):
    """projects into a valid correlation matrix"""
    if len(cov)<2: return cov,[],[]
    cor,sigma=cov2cor(cov)
    cor,e1,e2=truncate_eigenvalues_cor(cor,delta,checkpoint)
    cov=cor2cov(cor,sigma)
    return cov,e1,e2

def test_truncate_eigenvalues_cov():
    print "Testing truncate_eigenvalues_cov"
    n=5
    A=matrix(n,n)    
    for i in range(n):
        A[i][i]=abs(random.gauss(10,10))
        for j in range(i+1,n):
            A[i][j]=A[j][i]=random.gauss(10,10)
            pass
        pass
    print 'A=',
    pprint(A)
    B,e1,e2=truncate_eigenvalues_cov(A)
    print 'B=',
    pprint(B)
    for i in range(n):
        print e1[i],e2[i]
    return

def mean_LA(series):
    """series=[(datetime, rate),..]
    retuns mean and points (second col of series)"""
    sum=0.0
    rs=[]
    for d,r in series:
        rs.append(r)
        sum+=r
    return sum/len(series), rs
        
def covariance_LA(series1,m1,series2,m2):
    """series1=[(datetime, rate1),..], m1 is mean of rates1
    series2=[(datetime, rate2),..], m1 is mean of rates2
    retuns covariance and points"""
    sum=0.0
    rs=[]
    i=j=k=0
    while i<len(series1) and j<len(series2):
        d1,r1=series1[i]
        d2,r2=series2[j]
        if d1<d2: j+=1
        elif d1>d2: i+=1
        else:
            x=(r1-m1)*(r2-m2)
            sum+=x
            rs.append(x)
            i,j,k=i+1,j+1,k+1
            pass
        pass
    if k>0:
        return sum/k, rs
    return 0.0,[]

def test_all():
    test_AR1filter()
    test_truncate_eigenvalues_cov()


class Kalman:
    """ does not work yet """
    def __init__(self, ndim):
        self.ndim = ndim
        self.Sigma_x = identity(ndim,1e-5)
        self.A = identity(ndim)
        self.H = identity(ndim)
        self.mu_hat = 0
        self.cov = identity(ndim)
        self.R = identity(ndim,0.01)

    def update(self, obs):

        # Make prediction
        self.mu_hat_est = multiply(self.mu_hat,self.A)
        self.cov_est = add(multiply(self.A,multiply(self.cov,transpose(self.A))),
                           self.Sigma_x)
        # Update estimate
        self.error_mu = sub(obs,multiply(self.H,self.mu_hat_est))
        self.error_cov = add(multiply(self.H,multiply(self.cov,transpose(self.H))),
                             self.R)
        self.K = multiply(multiply(self.cov_est,transpose(self.H)),inverse(self.error_cov))
        self.mu_hat = add(self.mu_hat_est, multiply(self.K,self.error_mu))
        if ndim>1:
            self.cov = multiply(sub(identity(self.ndim),multiply(self.K,self.H)),self.cov_est)
        else:
            self.cov = multiply(sub(identity(self.ndim),self.K),self.cov_est)

def test_Kalman():
    print "***** 1d ***********"
    ndim = 1
    nsteps = 50
    k = Kalman(ndim)    
    mu_init=[-0.37727]
    cov_init=[0.1]*ndim
    obs = [[random.gauss(mu_init,cov_init) for i in range(nsteps)] for k in range(ndim)]
    for t in range(ndim,nsteps):
        k.update(obs[:,t])
    print k.mu_hat_est

    print "***** 2d ***********"
    ndim = 2
    nsteps = 50
    k = Kalman(ndim)    
    mu_init=array([-0.37727, 2.333])
    cov_init=[0.1]*ndim
    obs = matrix(ndim, nsteps)
    for t in range(nsteps):
        obs[:, t] = random.gauss(mu_init,cov_init)
    for t in range(ndim,nsteps):
        k.update(obs[:,t])
    print k.mu_hat_est

if __name__ == '__main__':
    import doctest
    test_all()
    # test_Kalman()
    doctest.testmod()

