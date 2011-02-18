import math

def erf(x):
     # save the sign of x
     sign = 1
     if x < 0:
          sign = -1
     x = abs(x)
     # constants
     a1 =  0.254829592
     a2 = -0.284496736
     a3 =  1.421413741
     a4 = -1.453152027
     a5 =  1.061405429
     p  =  0.3275911
     # A&S formula 7.1.26
     t = 1.0/(1.0 + p*x)
     y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
     return sign*y # erf(-x) = -erf(x)

def N(x):
    return 0.5+0.5*erf(x/math.sqrt(2))

def Call(S,T,K,r,sigma):
     d1 = (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
     d2 = (math.log(S/K)+(r-0.5*sigma**2)*T)/(sigma*math.sqrt(T))
     return N(d1)*S-N(d2)*K*math.exp(-r*T)

def f(x):
     return Call(134.0,30.0,133,0.04/300,x/math.sqrt(300))-5.3

from numeric import SolveNewtonBisection

x = SolveNewtonBisection(f,0.1,0.5) 
print x, f(x)
