from math import *
from random import *
from copy import copy, deepcopy
import time


################# Global Parameters #################

PRECISION=1e-6
EPSILON = 1e-3

############### Statistical Methods ###############

def mean(series):
    """Return the arithmetic mean of the series.

    Parameters
    ----------
    series : list/tuple
        One-dimensional number list/tuple, otherwise exception will be raised.

    Returns
    -------
    mean : float
 
    Examples
    --------
    >>> x=[1,2,3,4]
    >>> mean(x)
    2.5

    """ 
    return float(sum(series))/len(series)
    
def variance(series):
    """return the variance of the series.

    Parameters
    ----------
    series : list/tuple
        One-dimensional number list/tuple, otherwise exception will be raised.

    Returns
    -------
    variance : float
        
    See Also
    --------
    numeric.stddev

    Examples
    --------
    >>> x=[1,2,3,4]
    >>> variance(x)
    1.25
    """
    return mean([x**2 for x in series])-mean(series)**2

def stddev(series):
    """return the standard deviation of the series.

    Parameters
    ----------
    series : list/tuple
        One-dimensional number list/tuple, otherwise exception will be raised.

    Returns
    -------
    stddev : float
  
    See Also
    --------
    numeric.variance

    Examples
    --------
    @Modified by Sean Neilan
    1.118033988749895
    >>> x=[1,2,3,4]
    >>> round(stddev(x), 6)
    1.118034

    """
    return sqrt(variance(series))

def covariance(series_x, series_y):
    """return the covariance of the two series.

    Parameters
    ----------
    series_x : list
        One-dimensional number list/tuple, otherwise exception will be raised.
    series_y : list
        One-dimensional number list/tuple, otherwise exception will be raised.

    Returns
    -------
    covariance : float

    Note
    ----
    The lengths of the two series must be the same.
  
    See Also
    --------
    numeric.correlation

    Examples
    --------
    >>> x=[2,3,5,7]
    >>> y=[1,6,8,4]
    >>> covariance(x,y)
    1.8125

    """
    if len(series_x)!=len(series_y):
        raise RuntimeError, 'serieses have different size'
    return mean([x*series_y[i] for i,x in enumerate(series_x)])-mean(series_x)*mean(series_y)

def correlation(series_x, series_y):
    """return the correlation of the two series.

    Parameters
    ----------
    series_x : list
        One-dimensional number list/tuple, otherwise exception will be raised.
    series_y : list
        One-dimensional number list/tuple, otherwise exception will be raised.

    Returns
    -------
    correlation : float

    Note
    ----
    The lengths of the two series must be same, and each standard deviation
    must not be zero.
  
    See Also
    --------
    numeric.covariance

    Examples
    --------
    >>> x=[2,3,5,7]
    >>> y=[1,6,8,4]
    >>> correlation(x,y)
    0.36498927507141227

    """
    stdx=stddev(series_x)
    stdy=stddev(series_y)
    if stdx==0 or stdy==0:
        raise RuntimeError, 'zero standard deviation'
    return covariance(series_x,series_y)/(stddev(series_x)*stddev(series_y))


def bin(series,n):
    """Return the numbers of counts in each bin.

    Parameters
    ----------
    series : list
        1-dimensional number list.
    n : int
        Number(positive) of bins on the series, the interval woulb be equal.
        If n is not a integer, a conversion is attempted.

    Returns
    -------
    bins : list
        Return the numbers of counts in each bin.
    minimum : float
        The minimum of the series.
    maximum : float
        The maximum of the series.
    interval : float
        The interval of slices.
        
    Examples
    --------
    >>> bin([1,2,551,11,41,414,1224,1123,441,234],4)
    ([5, 3, 0, 2], 1.0, 1224.0, 305.75)

    """
    n=int(n)
    if n<=0 or len(series)==0:
        raise RuntimeError, 'no data'
    minim=float(min(series))
    maxim=float(max(series))
    interval=(maxim-minim)/n
    bins=[0]*n
    for x in series:
        if x!=maxim:
            bins[int((x-minim)/interval)]+=1
        else:
            bins[-1]+=1
    return bins,minim,maxim,interval

        
def E(f, series):
    """Return the expectation of the series given the function.

    Parameters
    ----------
    f : function
        The underlying function of the expectation.
    series : list/tuple
        The list/tuple to calculate the expectation.

    Return
    ------
    expectation : float
    
    Examples:
    ---------
    >>> print E(lambda X:X, [1,2,3,4,5])
    3.0
    >>> print E(lambda X:X, [[1],[2],[3],[4],[5]])
    3.0
    >>> print E(lambda X,Y,Z:X*Y*Z, [[1,1,2],[2,3,4],[3,0,2],[4,5,2],[5,0,0]])
    13.2

    """
    def to_tuple(x):
        return x if isinstance(x,(list,tuple)) else (x,)
    return float(sum(f(*to_tuple(a)) for a in series))/len(series)


def norm_1(A):
    """Calculates first norm of A. (The absolute total value of the
        largest column in A.)

    Parameters
    ----------
    A : matrix(N,M)

    Return
    ------
    m : double
        The first vector norm of A.

    Example
    --------
    >>> A = matrix(3,1)
    >>> A[0][0] = 1
    >>> A[1][0] = -1
    >>> A[2][0] = 2
    >>> print norm_1(A)
    4.0
    """
    z = m = 0.0
    for j in range(cols(A)):
        z = 0.0
        for i in range(rows(A)):
            z += abs(A[i][j])
        if z > m: m = z
    return m
    
        
############### Miscellaneous ###############

def prettylist(series):
    """Print the (1-dimensional number) series with 3 floating digits in string.

    Parameters
    ----------
    series : list
        1-dimensional number list.

    Returns
    -------
    prettylist : string
           
    Examples
    --------
    >>> x=[1,2.3,4.5513131,sin(2),2**12]
    >>> prettylist(x)
    '1.000,2.300,4.551,0.909,4096.000'

    """
    return ','.join(['%.3f' % x for x in series])
    
        
############### Linear Algebra ###############

def matrix(rows=1,cols=1):
    """Constuctor a all zero matrix.

    Parameters
    ----------
    rows : int
        Number of rows. A coversion is attempted if rows is not int. Absolute
        value is taken.
    cols : int
        Number of columns. A coversion is attempted if rows is not int.
        Absolute value is taken.
        
    Returns
    -------
    matrix : list
        Rows by cols matrix in list.
 
    Examples
    --------
    >>> matrix(3,4)
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    """
    return [[0]*int(cols) for k in range(int(rows))]


def pprint(A):
    """Print the matrix with four significant digits.

    Parameters
    ----------
    A : list
        The matrix to print.
 
    Examples
    --------
    >>> x=[[3.235,2,4.55],[14.5666,1.1234,23]]
    >>> pprint(x)
     [
      [ 3.235e+00, 2.000e+00, 4.550e+00, ],
      [ 1.457e+01, 1.123e+00, 2.300e+01, ],
     ]

    """
    print ' ['
    for line in A:
        print '  [',
        for item in line:
            print '%.3e,' % float(item),
            pass
        print '],'
        pass
    print ' ]'
    return
     
    
def rows(A):
    """Get the number of rows in the matrix.

    Parameters
    ----------
    A : list
        The matrix to get the rows.

    Returns
    -------
    rows : int
        Number of rows.
    
    Examples
    --------
    >>> x=matrix(3,5)
    >>> rows(x)
    3

    """
    return len(A)
         

def cols(A):
    """Get the number of columns in the matrix.

    Parameters
    ----------
    A : list
        The matrix to get the columns.

    Returns
    -------
    rows : int
        Number of columns.
    
    Examples
    --------
    >>> x=matrix(3,5)
    >>> cols(x)
    5

    """
    return len(A[0])


def add(A,B):
    """multiplies a number of matrix A by a matrix B.

    Parameters
    ----------
    A : int/float/list
        Either a single number or a matrix.

    Returns
    -------
    matrix : list
        The resulting matrix after addition.

    Examples
    --------
    >>> A=5
    >>> B=[[1,2], [3,4]]
    >>> add(A,B)
    [[6, 2], [3, 9]]

    """
    if type(A)==type(1) or type(A)==type(1.0):
        C=deepcopy(B)
        for i in range(rows(B)):            
            C[i][i]+=A
            pass
        return C
    else:
        C=deepcopy(B)
        for i in range(rows(B)):
            for j in range(cols(B)):
                C[i][j]+=A[i][j]
                pass
            pass
        return C
    pass


def sub(A,B):
    """multiplies a number of matrix A by a matrix B"""
    C=deepcopy(A)
    for i in range(rows(A)):
        for j in range(cols(A)): 
            C[i][j]-=B[i][j]
            pass
        pass
    return C


def multiply(A,B):
    """multiplies a number of matrix A by a matrix B"""
    if type(A)==type(1) or type(A)==type(1.0):
        C=matrix(rows(B),cols(B))
        for i in range(rows(B)):
            for j in range(cols(B)):
                C[i][j]=A*B[i][j]
        return C
    else:
        if cols(A) != rows(B):
            raise RuntimeError, 'matrices are different sizes'
        C=matrix(rows(A),cols(B))
        for i in range(rows(A)):
            for j in range(cols(B)):
                for k in range(cols(A)):
                    C[i][j]+=A[i][k]*B[k][j]
        return C


def test_multiply():
    """Test for the multiply(A,B) function"""
    print '\n\nTesting multiply(A,B)......... *integer*'
    A=5.0
    B=[[1,2], [3,4]]
    print 'A=',A
    print 'B=',B
    C=multiply(A,B)
    print 'C=',C
    print '\n\nTesting multiply(A,B)......... *matrix*'
    A=[[5, 6], [7, 8]]
    print 'A=',A
    print 'B=',B
    C=multiply(A,B)
    print 'C=',C
    print '\n\nTesting multiply(A,B)......... *fails*'
    try:
        A=[[1,2,3], [4,5,6], [7,8,9]]
        print 'A=',A
        print 'B=',B
        C=multiply(A,B)
        print 'C=',C
    except RuntimeError:
        print 'Expected failure. Continuing...'
    return

def inverse(A,checkpoint=None):
    """Computes the inverse of A using Gauss-Jordan elimination"""
    if rows(A) != cols(A):
        raise RuntimeError, 'matrix not squared'
    A=deepcopy(A)
    n=rows(A)
    B=matrix(n,n)
    for i in range(n): B[i][i]=1
    for c in range(n):
        if checkpoint: checkpoint('pivoting (%i) ...' % c)
        for r in range(c+1,n):
            if abs(A[r][c])>abs(A[c][c]):
                A[r],A[c],B[c],B[r]=A[c],A[r],B[r],B[c]
                pass
            pass
        p=float(A[c][c])
        for k in range(n):
            A[c][k],B[c][k]=float(A[c][k])/p,float(B[c][k])/p
            pass        
        for r in range(0,c)+range(c+1,n):
            p=float(A[r][c])
            for k in range(n):
                A[r][k]-=p*A[c][k]
                B[r][k]-=p*B[c][k]
                pass
            pass
        pass
    return B


def test_inverse():
    """Test for the inverse(A) function"""
    print '\n\nTesting inverse(A).........'
    A=[[1,2,3],[2,4,8],[1,3,7]]
    print 'A=',A
    B=inverse(A)
    print 'B=',B
    C=multiply(A,B)
    print 'A*B=',C
    try:
        print '\n\nTesting inverse(D)......... *fails*'
        D=[[1,2,3],[2,4,8]]
        print 'D=',D
        E=inverse(D)
        print 'E=',E
    except RuntimeError:
        print 'Expected failure. Continuing...'
    return

def transpose(A):
    """Transposed of A"""
    B=matrix(cols(A),rows(A))
    for i in range(rows(B)):
        for j in range(cols(B)):
            B[i][j]=A[j][i]
            pass
        pass
    return B

def Cholesky(A):
    if A!=transpose(A):
        raise RuntimeError, 'not symmetric'
    L=deepcopy(A)
    for k in range(cols(L)):
        if L[k][k]<=0:
            raise RuntimeError, 'not positive definitive'
        p=L[k][k]=sqrt(L[k][k])
        for i in range(k+1,rows(L)):
            L[i][k]/=p
            pass
        for j in range(k+1,rows(L)):
            p=float(L[j][k])
            for i in range(k+1,rows(L)):
                L[i][j]-=p*L[i][k]
                pass
            pass
        pass
    for  i in range(rows(L)):
        for j in range(i+1,cols(L)):
            L[i][j]=0
            pass
        pass
    return L

def test_Cholesky():
    """Test for the inverse(A) function"""
    print '\n\nTesting Cholesky(A).........'
    A=[[4,2,1],[2,9,3],[1,3,16]]
    print 'A=',A
    L=Cholesky(A)
    print 'L=',L
    C=sub(multiply(L,transpose(L)),A)
    print 'L*L^T-A=',C
    return

def Markoviz(mu, A, r_bar):
    """Assess Markoviz risk/return.

    Parameters
    ----------
    mu : list
        Matrix
    A : list
        Matrix
    r_bar : float
        Something

    Returns
    -------
    matrix : list
        Matrix
    
    Examples
    --------
    [[0.5566343042071197], [0.2750809061488674], [0.16828478964401297]]
    >>> A = matrix(3, 3)
    >>> A[0][0] = pow(0.20, 2)
    >>> A[1][1] = pow(0.30, 2)
    >>> A[2][2] = pow(0.40, 2)
    >>> A[0][1] = A[1][0] = 0.10 * 0.20 * 0.30
    >>> A[0][2] = A[2][0] = 0.25 * 0.20 * 0.40
    >>> A[1][2] = A[2][1] = 0.50 * 0.40 * 0.30
    >>> mu = matrix(3, 1)
    >>> mu[0][0] = 0.10
    >>> mu[1][0] = 0.12
    >>> mu[2][0] = 0.15
    >>> r_bar = 0.05
    >>> ret = Markoviz(mu, A, r_bar)
    >>> for m in range(len(ret)): ret[m][0] = round(ret[m][0], 7)
    >>> ret
    [[0.5566343], [0.2750809], [0.1682848]]
    """
    x = matrix(rows(A), 1)
    for r in range(rows(mu)): mu[r][0] -= r_bar
    x = multiply(inverse(A), mu)
    x_norm = 0.0
    for r in range(rows(mu)): x_norm += x[r][0]
    for r in range(rows(mu)): x[r][0] /= x_norm
    return x


def test_Markoviz():
    """Test for the Markoviz(mu, A, r_bar) function"""
    A = matrix(3, 3)
    A[0][0] = pow(0.20, 2)
    A[1][1] = pow(0.30, 2)
    A[2][2] = pow(0.40, 2)
    A[0][1] = A[1][0] = 0.10 * 0.20 * 0.30
    A[0][2] = A[2][0] = 0.25 * 0.20 * 0.40
    A[1][2] = A[2][1] = 0.50 * 0.40 * 0.30
    mu = matrix(3, 1)
    mu[0][0] = 0.10
    mu[1][0] = 0.12
    mu[2][0] = 0.15
    r_bar = 0.05
    
    x = Markoviz(mu, A, r_bar)
    
    pprint(x)
    print 'return: ', multiply(transpose(mu), x)
    print 'risk: ', sqrt(multiply(multiply(transpose(x), A), x)[0][0])


def identity(n):
    A=matrix(n,n)
    for i in range(n): A[i][i]=1
    return A

def diagonal(v):
    n=len(v)
    A=matrix(n,n)
    for i in range(n): A[i][i]=v[i]
    return A

def maxind(S,k):
    j=k+1
    for i in range(k+2,len(S[k])):
        if abs(S[k][i])>abs(S[k][j]): j=i
        pass
    return j

def Jacobi(A,checkpoint=False):
    """Returns U end e so that A=U*diagonal(e)*transposed(U)
       where i-column of U contains the eigenvector corresponding to
       the eigenvalue e[i] of A.

       from http://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm

    """
    t0=time.time()
    n=rows(A)
    if n!=cols(A):
        raise RuntimeError, 'matrix not squared'
    S=matrix(n,n)
    for i in range(n):
        for j in range(n):
            S[i][j]=float(A[i][j])
            pass
        pass
    E=identity(n)
    state=n
    ind=[maxind(S,k) for k in range(n)]
    e=[S[k][k] for k in range(n)]
    changed=[True for k in range(n)]
    iteration=0
    while state:
        if checkpoint: checkpoint('rotating vectors (%i) ...' % iteration)
        m=0
        for k in range(1,n-1):
            if abs(S[k][ind[k]])>abs(S[m][ind[m]]): m=k
            pass
        k,h=m,ind[m]
        p=S[k][h]
        y=(e[h]-e[k])/2
        t=abs(y)+sqrt(p*p+y*y)
        s=sqrt(p*p+t*t)
        c=t/s
        s=p/s
        t=p*p/t
        if y<0: s,t=-s,-t
        S[k][h]=0
        #update(k,-t)
        y=e[k]
        e[k]=y-t
        if changed[k] and y==e[k]: changed[k],state=False,state-1
        elif (not changed[k]) and y!=e[k]: changed[k],state=True,state+1
        #update(h,t)
        y=e[h]
        e[h]=y+t
        if changed[h] and y==e[h]: changed[h],state=False,state-1
        elif (not changed[h]) and y!=e[h]: changed[h],state=True,state+1
        
        for i in range(k):
            S[i][k],S[i][h]=c*S[i][k]-s*S[i][h],s*S[i][k]+c*S[i][h]
            pass
        for i in range(k+1,h):
            S[k][i],S[i][h]=c*S[k][i]-s*S[i][h],s*S[k][i]+c*S[i][h]
            pass
        for i in range(h+1,n):
            S[k][i],S[h][i]=c*S[k][i]-s*S[h][i],s*S[k][i]+c*S[h][i]
            pass
        for i in range(n):
            E[k][i],E[h][i]=c*E[k][i]-s*E[h][i],s*E[k][i]+c*E[h][i]
            pass
        ind[k],ind[h]=maxind(S,k),maxind(S,h)
        iteration+=1
        pass
    # SORT VECTORS
    for i in range(1,n):
        j=i
        while j>0 and e[j-1]>e[j]:
            e[j],e[j-1]=e[j-1],e[j]
            E[j],E[j-1]=E[j-1],E[j]
            j-=1
            pass
        pass
    # NORMALIZE VECTORS
    U=matrix(n,n)
    for i in range(n):
        sum=0.0
        for j in range(n): sum+=E[i][j]**2;            
        sum=sqrt(sum)
        for j in range(n): U[j][i]=E[i][j]/sum
        pass
    return U,e
    

def test_Jacobi():
    """Test the Jacobi algorithm"""
    print 'Testing Jacobi on random matrices...'
    n=4
    A=matrix(n,n)
    for k in range(3):
        for i in range(n):
            for j in range(i,n):
                A[i][j]=A[j][i]=gauss(10,10)
                pass
            pass
        print 'A=',
        pprint(A)
        U,e=Jacobi(A)
        print 'U*e*U^T-A=',
        pprint(sub(multiply(U,multiply(diagonal(e),transpose(U))),A))
        pass
    return

def fitting_function(f,C,x):
    sum=0.0
    i=0
    for func in f:
        sum+=func(x)*C[i][0]
        i+=1
        pass
    return sum
    

def fit(f,x,y,dy=None):
    """Linear fit of y[i]+/idy[i] using sum_j f[j](x[i])"""
    A=matrix(len(x),len(f))
    B=matrix(len(y),1)
    for i in range(rows(A)):
        w=1.0
        if dy: w=1.0/sqrt(dy[i])
        B[i][0]=w*y[i]
        for j in range(cols(A)):
            A[i][j]=w*f[j](x[i])
            pass
        pass
    C=multiply(inverse(multiply(transpose(A),A)),multiply(transpose(A),B))
    chi=sub(multiply(A,C),B)
    chi2=multiply(transpose(chi),chi)
    ff=lambda x,C=C,f=f,q=fitting_function: q(f,C,x)
    return C,chi2[0][0],ff

"""Sets of fitting functions for fit"""
CONSTANT=[1]
LINEAR=[lambda x: 1.0, lambda x: x]
QUADRATIC=[lambda x: 1.0, lambda x: x, lambda x: x*x]
CUBIC=[lambda x: 1.0, lambda x: x, lambda x: x*x, lambda x: x*x*x]
QUARTIC=[lambda x: 1.0, lambda x: x, lambda x: x*x, lambda x: x*x*x, lambda x: x*x*x*x]
def POLYNOMIAL(n):
    """Generic polynmial fitting function"""
    return [(lambda x,i=i: pow(x,i)) for i in range(n+1)]
def EXPONENTIAL(n):
    """Generic exponential fitting function"""
    return [(lambda x,i: exp(x*i)) for i in range(n+1)]

def test_fit():
    """Test for the fit function"""
    print '\n\nTesting fit(QUADRATIC,...).........'
    print 'data generated using 5+0.8*k+0.3*k*k+gauss(0,1)'
    x=[]
    y=[]
    for k in range(100):
        x.append(k)
        y.append(5+0.8*k+0.3*k*k+gauss(0,1))
        pass
    a,chi2,ff=fit(QUADRATIC,x,y)
    print 'f(x)=(',a[0][0],')+(',a[1][0],')*x+(',a[2][0],')*x*x'
    print 'chi2=',chi2
    
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
    print '\n\nTesting AR1filter().........'
    y=[(0,0.05)]
    for k in range(1,30):
        y.append((k,0.05+y[k-1][1]*0.3+gauss(0,0.01)))
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
    print 'Testing truncate_eigenvalues_cov'
    n=5
    A=matrix(n,n)
    for i in range(n):
        A[i][i]=abs(gauss(10,10))
        for j in range(i+1,n):
            A[i][j]=A[j][i]=gauss(10,10)
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


############## new stuff from class Function ##############
class NoConvergence(RuntimeError):
    pass
class WrongParameters(RuntimeError):
    pass

############## numerical derivatives ##############
def D(f,h=EPSILON):
    return (lambda x,f=f,h=h: (f(x+h)-f(x-h))/(2.0*h))

def D2(f,h=EPSILON):
    return (lambda x,f=f,h=h: (f(x+h)-2.0*f(x)+f(x-h))/(h*h))


############## Solvers and Optimizers ############        

def SolveFixedPoint(f, x_guess):
    """Find a point where x = f(x) intersects the line y = x.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    x_guess : double
        Starting point for search

    Returns
    -------
    x : double
        Point where f(x) = x
 
    Example
    --------
    >>> f = lambda x: (x+7)/(x-4)
    >>> x = 1.0
    >>> str(SolveFixedPoint(f, x))[0:8]
    '-6.99999'

    """ 

    def g(x): return f(x) + x

    x = x_guess
    x_old = x + 2.0 * PRECISION

    while abs(x_old-x)>=PRECISION:
        if abs(D(g)(x)) >= 1: 
            raise RuntimeError, 'no convergence'
        x_old=x
        x = g(x)
    return x


def SolveBisection(f, a, b):
    """Finds a point x between a and b such that f(x) == 0.

    Parameters
    ----------
    f : function
        Function taking one parameter x.
    a : double
        A double such that f(a) is negative if f(b) is positive or vice versa.
    b : double
        A double such that f(b) is negative if f(a) is positive or vice versa.

    Returns
    -------
    x : double
        Point between a and b such that f(x) == 0.
 
    Example
    --------
    >>> f = lambda x: (x+4)*(x-3)*(x+7)
    >>> a = 0.0
    >>> b = 4.0
    >>> SolveBisection(f, a, b)
    3.0
    """ 

    fa = f(a)
    fb = f(b)
    x = fx = 0.0
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0:
        raise RuntimeError, 'f(a) and f(b) must have opposite sign'

    for k in range(20):
        #print x
        x = (a+b)/2
        fx = f(x)
        if abs(fx) < PRECISION:
            return x
        elif fx * fa < 0:
             b = x
             fb = fx
        elif fx * fb < 0:
            a = x
            fa = fx
    return x


def OptimizeBisection(f, a, b):
    """Finds a point x between a and b such that f'(x) == 0
    f'(a) must be negative, f'(b) must be positive or vice versa.

    Parameters
    ----------
    f : function
        Function taking one parameter x.
    a : double
        A double such that f'(a) is negative if f'(b) is positive or vice versa.
    b : double
        A double such that f'(b) is negative if f'(a) is positive or vice versa.

    Returns
    -------
    x : double
        Point between a and b such that f'(x) == 0.
 
    Example
    --------
    >>> f = lambda x: pow(x,4)/4+(8.0*pow(x,3))/3-(5.0*pow(x,2))/2-84.0*x
    >>> a = 0.0
    >>> b = 4.0
    >>> str(OptimizeBisection(f, a, b))[0:5]
    '2.999'
    """

    f1a = D(f)(a)
    f1b = D(f)(b)
    x = fx = 0.0
    if f1a == 0: return a
    if f1b == 0: return b
    if f1a*f1b > 0:
        raise RuntimeError, 'D(f)(a) and D(f)(b) must have opposite parameters'

    for k in range(20):
        #print x
        x = (a+b)/2
        f1x = D(f)(x)
        if abs(f1x) < PRECISION:
            return x
        elif f1x*f1a<0:
            b = x
            f1b = f1x
        elif f1x*f1b<0:
            a = x
            f1a = f1x
    return x



def SolveNewton(f, x):
    """Starts with a guess x that is reasonably close to a root of f(x). Uses 
        Newton's iteration to find this root.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    x : double
        Starting point for search

    Returns
    -------
    x : double bubble
        Point where f(x) = x
 
    Example
    --------
    Find the square root of 612
    x^2 = 612
    f(x) = pow(x, 2)-612
    initial guess of 10
    >>> f = lambda x: pow(x,2)-612
    >>> x = 10.0
    >>> str(SolveNewton(f, x))[0:9]
    '24.738633'
    """
    x=float(x)
    x_old = x+PRECISION
    f1x = 0.0
    for k in range(20):
        #print x
        f1x = D(f)(x)
        if abs(f1x) < PRECISION:
            raise RuntimeError, 'unstable solution'
        x_old = x
        x = x-f(x)/f1x
        if abs(x-x_old) < PRECISION: return x
    raise RuntimeError, 'no convergence'


def OptimizeNewton(f, x):
    """Starts with a guess x that is reasonably close to a root of f'(x). Uses 
        Newton's iteration to find this root.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    x : double
        Starting point for search

    Returns
    -------
    x : double
 
    Example
    --------
    >>> f = lambda x: (2*x+9)*(pow(x,2)-6)
    >>> x = .5
    >>> str(OptimizeNewton(f, x))[0:8]
    '0.561552'
    """
    x=float(x)
    x_old = x + PRECISION
    f2x = 0.0
    for k in range(20):
        #print x
        f2x = D2(f)(x)
        if abs(f2x)<PRECISION:
            raise RuntimeError, 'unstable solution'
        x_old = x
        x = x-D(f)(x)/f2x
        if abs(x-x_old) < PRECISION: return x
    raise RuntimeError, 'no covergence'


def SolveNewtonBisection(f, a, b):
    """Finds a number c such that a < c < b and 
    f(x) = 0. Uses Newton's iteration to find this root.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    a : double
        A double such that f(a) is negative if f(b) is positive or vice versa.
    b : double
        A double such that f(b) is negative if f(a) is positive or vice versa.

    Returns
    -------
    x : double
        Point between a and b such that f(x) == 0.
 
    Example
    --------
    >>> f = lambda x: (2*x+9)*(pow(x,2)-6)
    >>> a = -5
    >>> b = 5
    >>> str(SolveNewtonBisection(f, a, b))[0:8]
    '2.449489'

    """
    a,b=float(a),float(b)
    fa = f(a)
    fb = f(b)
    x = fx = 0.0
    if fa == 0: return a
    if fb == 0: return b
    if fa*fb > 0:
        raise RuntimeError, 'f(a) and f(b) must have opposite sign'
    f1x = 0.0
    for k in range(20):
        #print x
        if abs(f1x) > PRECISION:
            x = x - fx/f1x
        if abs(f1x) <= PRECISION or x <= a or x >= b:
            x = (a + b)/2
        fx = f(x)
        f1x = D(f)(x)
        if abs(fx) < PRECISION:
            return x
        elif fx * fa < 0:
            b = x
            fb = fx
        elif fx * fb < 0:
            a = x
            fa = fx
    return x


def OptimizeNewtonBisection(f, a, b):
    """Finds a number c such that a < c < b and 
    f'(x) = 0. Uses Newton's iteration to find this root.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    a : double
        A double such that f'(a) is negative if f'(b) is positive or vice versa.
    b : double
        A double such that f'(b) is negative if f'(a) is positive or vice versa.

    Returns
    -------
    x : double
        Point between a and b such that f'(x) == 0.
 
    Example
    --------
    >>> f = lambda x: (2*x+9)*(pow(x,2)-6)
    >>> a = -2
    >>> b = 1
    >>> str(OptimizeNewtonBisection(f, a, b))[0:8]
    '0.561552'
    """
    a,b=float(a),float(b)
    f1a = D(f)(a)
    f1b = D(f)(b)
    if f1a == 0: return a
    if f1b == 0: return b
    if f1a * f1b > 0:
        raise RuntimeError, 'D(f)(a) and D(f)(b) must have opposite sign'
    x = (a+b)/2
    f1x = D(f)(x)
    f2x = D2(f)(x)
    for k in range(20):
        #print x
        if abs(f1x) > PRECISION:
            x = x - f1x/f2x
        if abs(f2x) <= PRECISION or x <= a or x >= b:
            x = (a + b) / 2
        f1x = D(f)(x)
        f2x = D2(f)(x)
        if abs(f1x) < PRECISION:
            return x
        elif f1x * f1a < 0:
            b = x
            f1b = f1x
        elif f1x * f1b < 0:
            a = x
            f1a = f1x
    return x


def SolveSecant(f, x):
    """Starts with a guess x that is reasonably close to a root of f(x). Uses 
        the Secant Method iteration to find this root. Returns a number
        x such that f(x) = 0.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    x : double
        Starting point for search

    Returns
    -------
    x : double
 
    Example
    --------
    >>> f = lambda x: cos(x) - x * x * x
    >>> x = 1.0
    >>> str(SolveSecant(f, x))[0:8]
    '0.865474'
    """
    x=float(x)
    x_old = 0.0
    fx = f1x = f_old = 0.0
    x_old = x - 0.0001
    f_old = f(x_old)

    for k in range(20):
        #print x
        fx = f(x)
        f1x = (fx-f_old)/(x-x_old)
        if abs(f1x) < PRECISION:
            raise RuntimeError, 'instability'
        f_old = fx
        x_old = x
        x = x - fx/f1x
        if k > 1 and abs(x-x_old)<PRECISION: return x
    raise RuntimeError, 'no convergence'




def OptimizeSecant(f, x):
    """Starts with a guess x that is reasonably close to a root of f'(x). Uses 
        the Secant Method iteration to find this root.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    x : double
        Starting point for search

    Returns
    -------
    x : double
 
    Example
    --------
    >>> f = lambda x: (x-3)**2
    >>> x = 1.0
    >>> str(OptimizeSecant(f, x))[0:8]
    '3.0'
    """
    x=float(x)
    x_old = f1x = f2x = f1_old = 0.0
    x_old = x - 0.0001
    f1_old = D(f)(x_old)
    for k in range(20):
        #print x
        f1x = D(f)(x)
        f2x = (f1x-f1_old)/(x-x_old)
        if abs(f2x)<PRECISION:
            raise RuntimeError, 'unstable solution'
        f1_old = f1x
        x_old = x
        x = x-f1x/f2x
        if k > 1 and abs(x-x_old) < PRECISION: 
            return x
    raise RuntimeError, 'no convergence'


def OptimizeGoldenSection(f, a, b):
    """ Searches for the minimum of a given function, f, with
    the initial interval being between [a, b].
    
    Parameters
    ----------
    f : function
        Function taking one parameter a or b
    a : double
        The lower-bound of the initial interval
    b : double
        The upper-bound of the initial interval
    
    Returns
    -------
    double : The minimum of the function
    
    Example
    -------
    >>> f = lambda x: x
    >>> a = 5.123876123
    >>> b = 14.182
    >>> str(OptimizeGoldenSection(f, a, b))[0:8]
    '5.123876'
    """
    a,b=float(a),float(b)
    t = (sqrt(5.0)-1.0)/2.0
    x1 = a+(1.0-t)*(b-a)
    x2 = a+(t)*(b-a)
    fa = f(a)
    fb = f(b)
    f1 = f(x1)
    f2 = f(x2)

    while abs(b-a) > PRECISION:
        #print (x1-a)/(b-a)
        if f1 > f2:
            a = x1
            fa = f1
            x1 = x2
            f1 = f2
            x2 = a+(t)*(b-a)
            f2 = f(x2)
        else:
            b = x2
            fb = f2
            x2 = x1
            f2 = f1
            x1 = a + (1.0-t)*(b-a)
            f1 = f(x1)
    return b


def OptimizeGoldenSection2(f, a, b, t=0.8):
    """ Searches for the minimum of a given function, f, with
    the initial interval being between [a, b]. With
    optional tau, t, value.
    
    Parameters
    ----------
    f : function
        Function taking one parameter a or b
    a : double
        The lower-bound of the initial interval
    b : double
        The upper-bound of the initial interval
    t : double (default: 0.8)
        Optional tau value
    
    Returns
    -------
    double : The minimum of the function with tau value, t
    
    Example
    -------
    >>> f = lambda x: x
    >>> a = 7.00005
    >>> b = 128.129876239
    >>> str(OptimizeGoldenSection2(f, a, b))[0:8]
    '7.000050'
    """
    a,b= float(a),float(b)
    x1 = a + (1.0-t)*(b-a)
    x2 = a + t*(b-a)
    fa = f(a)
    fb = f(b)
    f1 = f(x1)
    f2 = f(x2)
    while abs(b-a)>PRECISION:
        #print (x1-a)/(b-a)
        if f1>f2:
            a = x1
            fa = f1
            x1 = x2
            f1 = f2
            x2 = x1+t*(b-x1)
            f2 = f(x2)
        else:
            b = x2
            fb = f2
            x2 = x1
            f2 = f1
            x1 = a + (1.0-t)*(x2-a)
            f1 = f(x1)
    return b


def IntegrateNaive(f, a, b):
    """ Integrates function, f, from a to b using the Reimann sum.
    
    Parameters
    ----------
    f : function
        Function to be integrated
    a : double
        Lower-bound of integration
    b : double
        Upper-bound of integration
    
    Returns
    -------
    double
    
    Example
    -------
    >>> f = lambda x: x
    >>> a = 3.0
    >>> b = 4.0
    >>> IntegrateNaive(f, a, b)
    3.4999990463256836
    """
    a,b= float(a),float(b)
    I = h = Iold = 0.0
    N = 2
    while True:
        # print N
        Iold = I
        I = 0.0
        h = (b-a)/N # the width of each Reimann rect
        for i in range(N):
            I += h*f(a+i*h) # I contains the area of each Reimann rect
        if N != 2 and abs(I-Iold) < PRECISION: return I # keep running this function until we don't get much change in value
        N *= 2
    return I


def IntegrateNaive2(f, a, b):
    """ Integrates function, f, from a to b using Reimann sum, except,
        last and first rectangles have a width of 0.5.
    
    Parameters
    ----------
    f : function
        Function to be integrated
    a : double
        Lower-bound of integration
    b : double
        Upper-bound of integration
    
    Returns
    -------
    double
    
    Example
    -------
    >>> f = lambda x: x
    >>> a = 3.0
    >>> b = 4.0
    >>> IntegrateNaive2(f, a, b)
    3.5
    """
    a,b= float(a),float(b)
    I = h = Iold = hold = 0.0
    N = 2
    while True:
        #print N
        Iold = I
        hold = h
        I = 0.5 * f(a)
        h = (b-a)/N
        for i in range(1, N):
            I += f(a+i*h)
        I += 0.5 * f(b)
        #print N, ' ', I
        #print I * h
        #print str(abs(I*h-Iold*hold))[0:8]
        #print I*h, Iold*hold
        if N != 2 and abs(I*h-Iold*hold) < PRECISION: return h*I
        #if N != 2 and abs(I*h-Iold*h) < PRECISION: return h*I
        N *= 2


def IntegrateQuadrature(f, a, b, N):
    """Calculates the integral of the function f from points a to b with N Vandermonde weights
        using numerical quadrature.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    a : double
        Value to start calculating the integral on
    b : double
        Value to stop calculating on
    N : integer
        Number of squares used to estimate the integral.

    Returns
    -------
    I : double
        The estimated area under f between [a,b].
 
    Example
    --------
    >>> f = lambda x: (x+4)*(x-5)
    >>> str(IntegrateQuadrature(f, 0.0, 10.0, 5))[0:8]
    '83.33333'

    Notes
    --------
    Blows up for large N
    """
    a,b= float(a),float(b)
    h = (b-a)/N

    # create a matrix
    A = matrix(N, N)
    C = matrix(N)
    W = matrix(N)
    for i in range(N):
        for j in range(N):
            A[i][j] = pow(a+(j+1)*h,i)
        C[i][0] = (pow(b, i+1)-pow(a,i+1))/(i+1)
    W = multiply(inverse(A), C)
    blah = multiply(A, W)
    I = 0.0
    for i in range(N):
        I += W[i][0]*f(a+(i+1)*h)
    return I


def IntegrateQuadrature2(f, a, b):
    """Computes the most precise value we can get out of
        IntegrateQuadrature for the values of f, a and b.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    a : double
        Value to start calculating the integral on
    b : double
        Value to stop calculating on

    Returns
    -------
    I : integer
        The most precise return value from IntegrateQuadrature.
        
    Example
    --------
    >>> f = lambda x: (x+4)*(x-5)
    >>> str(IntegrateQuadrature2(f, 0.0, 10.0))[0:8]
    '83.33333'

    Notes
    --------
    Blows up for large N
    Due to imprecision between C libraries on different systems,
        the test case is limited to 8 characters.
    """
    a,b= float(a),float(b)
    I = Iold = 0.0
    n = 2
    while True:
        #print n
        Iold = I
        I = IntegrateQuadrature(f, a, b, n)
        #print 'I=', I
        if n > 2 and abs(I-Iold)<PRECISION: return I
        n *= 2


def IntegrateAdaptativeQuadrature(f, a, b, n1=3, n2=4):
    """Recursively calculates smaller and smaller integrals
        in the area under f(x) in [a,b]. Attempts to reach
        a maximum level of precision in each integral part.

    Parameters
    ----------
    f : function
        Function taking one parameter x
    a : double
        Value to start calculating the integral on
    b : double
        Value to stop calculating on
    n1 : integer
        Lower bound on squares to calculate integral
    n2 : integer
        Upper bound on squares to calculate integral

    Returns
    -------
    I : integer
        The most precise return value from IntegrateQuadrature.
        
    Example
    --------
    >>> f = lambda x: (x+4)*(x-5)
    >>> str(IntegrateAdaptativeQuadrature(f, 0.0, 10.0))[0:8]
    '83.33333'

    Notes
    --------
    Blows up for large N
    Due to imprecision between C libraries on different systems,
        the test case is limited to 8 characters.
    """
    a,b= float(a),float(b)
    I1 = I2 = 0.0
    try:
        I1 = IntegrateQuadrature(f, a, b, n1)
        I2 = IntegrateQuadrature(f, a, b, n2)
    except RuntimeError:
        raise RuntimeError, 'n2 is too large'
    m = a+(b-a)/2.0
    if m <= a or m >= b:
        print 'warning - possible precision problem'
        return I2
    if abs(I1-I2)<PRECISION: return I2
    return IntegrateAdaptativeQuadrature(a, m, n1, n2) + \
           IntegrateAdaptativeQuadrature(m, b, n1, n2)



### multi dimensional function utilities TO DO FROM NOW ON

def Jacobian(f, x):
    """Returns the first derivative of a vector function of several variables at 
        point x.

    Parameters
    ----------
    f : function
        Function taking a column matrix x
    x : Single column matrix, matrix(n,1)
        Point at which to find derivative of f at.

    Returns
    -------
    J : matrix(rows(x), 1)
        The derivative of f at point x.

    Example
    --------
    >>> def f(x):
    ...     y = matrix(rows(x),1)
    ...     y[0][0] = x[0][0]*x[1][0]*x[2][0]+8
    ...     y[1][0] = x[1][0]-x[0][0]-3.0*x[2][0]
    ...     y[2][0] = x[0][0]*x[2][0]+x[1][0]
    ...     return y
    >>> x = matrix(3,1)
    >>> x[0][0] = 1
    >>> x[1][0] = -1
    >>> x[2][0] = 2
    >>> pprint(Jacobian(f, x))
     [
      [ -2.000e+00, 2.000e+00, -1.000e+00, ],
      [ -1.000e+00, 1.000e+00, -3.000e+00, ],
      [ 2.000e+00, 1.000e+00, 1.000e+00, ],
     ]
    """

    h = EPSILON
    n = rows(x)
    J = matrix(n,n)
    A = matrix(n,1)
    #x_plus_h = x(n,1)
    x_plus_h = deepcopy(x)
    for j in range(n):
        if j > 0: x_plus_h[j-1][0] -= h
        x_plus_h[j][0] += h
        A = sub(f(x_plus_h), f(x))
        for i in range(n):
            J[i][j] = A[i][0]/h
    return J


def fix(h):
    """This scales the matrix down if necessary.
        If any values in the column matrix h are larger than 10.0,
        fix will multiply every value by 10.0/n where n is the largest
        absolute value in h. 

    Parameters
    ----------
    h : matrix(n,1)
        This matrix will be scaled down if necessary

    Returns
    -------
    Nothing. h is modified as a side effect.

    Example
    --------
    >>> h = matrix(3,1)
    >>> h[0][0] = 3
    >>> h[1][0] = -4
    >>> h[2][0] = 11
    >>> fix(h)
    >>> pprint(h)
     [
      [ 2.727e+00, ],
      [ -3.636e+00, ],
      [ 1.000e+01, ],
     ]
    """
    max_jump = 10.0
    m = 0.0
    for i in range(rows(h)):
        if abs(h[i][0]) > m: m = float(abs(h[i][0]))
    if m > max_jump:
        for i in range(rows(h)):
            h[i][0] *= max_jump/m


def SolveNewtonForMultiDimFunc(f, x):
    """Computes the root of a multidimensional function f near point x.

    Parameters
    ----------
    f : function
        f should take a column matrix and return a column matrix
    x : matrix(N,1)
        A point to start searching for the root of f

    Returns
    -------
    x : matrix(N,1)
        A root of f such that f(x) returns 0's.

    Example
    --------
    >>> def f(x):
    ...     y = matrix(rows(x),1)
    ...     y[0][0] = x[0][0]*x[1][0]*x[2][0]+8
    ...     y[1][0] = x[1][0]-x[0][0]-3.0*x[2][0]
    ...     y[2][0] = x[0][0]*x[2][0]+x[1][0]
    ...     return y
    >>> x = matrix(3,1)
    >>> x[0][0] = 1
    >>> x[1][0] = -1
    >>> x[2][0] = 2
    >>> pprint(x)
     [
      [ 1.000e+00, ],
      [ -1.000e+00, ],
      [ 2.000e+00, ],
     ]
    >>> x = SolveNewtonForMultiDimFunc(f, x)
    >>> pprint(x)
     [
      [ 4.652e+00, ],
      [ 2.828e+00, ],
      [ -6.080e-01, ],
     ]
    >>> pprint(f(x)) #@TODO this is incorrent. first value should be 3.38271e-16 due to precision problems in f(x) between c++/python
     [
      [ 0.000e+00, ],
      [ 2.220e-16, ],
      [ -4.441e-16, ],
     ]
    
    """
    x_old = matrix(rows(x), 1)
    J = matrix(rows(x), rows(x))
    h = matrix(rows(x), 1)
    for k in range(100):
        J = Jacobian(f, x)
        #pprint(x)
        #print 'f(x) round ', k, ' is ', f(x)
        #print 'jacobian round ', k, ' is ', Jacobian(f,x)
        if norm_1(J) < PRECISION:
            raise RuntimeError, 'unstable solution'
        x_old = deepcopy(x)
        h = multiply(inverse(J), f(x))
        fix(h)
        x = sub(x, h)
        #print 'x round ', k, ' is ', x
        #print 'f(x) round ', k, ' is ', f(x)
        if norm_1(sub(x,x_old)) < PRECISION: return x
    raise RuntimeError, 'no convergence'


### FunctionOfMultipleVariables stuff
"""
	double f(double alpha) {
		Matrix x=x_sd+alpha*s_sd;
		return f(x);
	}
	Matrix gradient(const Matrix& x) {
		double h=EPSILON;
		int n=x.nrows;
		Matrix v(n);
    		Matrix x_plus_h=x; 
	        for(int j=0; j<n; j++) {
	            x_plus_h(j)+=h;
		    v(j)=(f(x_plus_h)-f(x))/h;
		    x_plus_h(j)-=h;
    		}		
		return v;
	}
	Matrix Hessian(const Matrix& x) {
		double h=EPSILON;
		int n=x.nrows;
                Matrix H(n,n);
		Matrix x_plus_hi=x; 
		Matrix x_plus_hi_minus_hj=x;
		Matrix x_minus_hj=x; 
		double tmp;
                for(int i=0; i<n; i++) {
      			x_plus_hi(i)+=h;	
      			x_plus_hi_minus_hj(i)+=h;
			for(int j=0; j<n; j++) {
	      			x_plus_hi_minus_hj(j)-=h;
				x_minus_hj(j)-=h;
				H(i,j)=(f(x_plus_hi)-f(x_plus_hi_minus_hj)-f(x)+f(x_minus_hj))/(h*h);
	      			x_plus_hi_minus_hj(j)+=h;
				x_minus_hj(j)+=h;				
			}
      			x_plus_hi(i)-=h;	
      			x_plus_hi_minus_hj(i)-=h;
		   }
		return H;
	}
	Matrix OptimizeSteepestDescent(const Matrix& x0) {
		x_sd=x0;
		Matrix x_old=x_sd;
		double alpha;
		for(int k=0; k<100; k++) {				
			s_sd=gradient(x_sd);			
			try {
				alpha=OptimizeSecant(0.0);
			} catch(Exception e) {
			  cout << 'oops, not sure if converges!\n';
				break;
			}
			x_old=x_sd;
			x_sd=x_sd+alpha*s_sd;
			if(norm_2(x_sd-x_old)<PRECISION) break;
		}
		return x_sd;
	}

  Matrix OptimizeNewton(Matrix x) {
    Matrix x_old;
    Matrix g;
    Matrix H;
    for(int k=0; k<100; k++) {
      cout << x << endl;
      g=gradient(x);
      H=Hessian(x);
      if(norm_1(H)<PRECISION) throw Exception('Instability');
      x_old=x;
      x=x-inverse(H)*g;	
      if(norm_1(x-x_old)<PRECISION) return x;
    }
    throw Exception('NoCovergence');
  }
  Matrix OptimizeNewtonWithLinearContrants(Matrix x, Matrix B, Matrix d) {
	// contraint is Bx-d=0
    Matrix x_old;
    Matrix b;
    Matrix A;
    for(int k=0; k<100; k++) {
      cout << x << endl;
      b=gradient(x);
      A=Hessian(x);
      if(norm_1(A)<PRECISION) throw Exception('Instability');
      x=OptimizerForQuadraticFunctionWithLinearConstraints(A,b,B,d);
      if(norm_1(x-x_old)<PRECISION) return x;
    }
    throw Exception('NoCovergence');
  }
"""


def test_all():
    test_multiply()
    test_inverse()
    test_Cholesky()
    test_Markoviz()
    test_fit()
    test_AR1filter()
    test_Jacobi()
    test_truncate_eigenvalues_cov()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
