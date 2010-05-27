from yahoo import *
from math import *
import sys
import random
from mc import *
# time is seconds



class OptionPricerSimple(MCSimulator):

    def __init__(self,p0,log_returns,days,s,
                 r_free=0.02,option_type=None):
        self.p0=p0
        self.log_returns=log_returns
        self.days=days
        self.s=s
        self.r_free=r_free
        self.option_type=option_type

    def simulate_once(self):
        daily_returns=[]
        for t in range(self.days):
            i=random.randint(0,len(log_returns)-1)
            daily_returns.append(self.log_returns[i])
        return self.option_type(daily_returns)

    def european_put(self,daily_returns):
        r=sum(daily_returns)
        p = exp(r)*self.p0
        print 'expiration_price=',p
        return max(self.s-p,0)*exp(-self.r_free*self.days/250)

    def european_call(self,daily_returns):
        r=sum(daily_returns)
        p = exp(r)*self.p0
        return max(p-self.s,0)*exp(-self.r_free*self.days/250)

    def digital_put(self,daily_returns):
        r=sum(daily_returns)
        p = exp(r)*self.p0
        if self.s>p: return 1.0
        return 0.0

    def digital_call(self,daily_returns):
        r=sum(daily_returns)
        p = exp(r)*self.p0
        if p>self.s:
            return 1.0
        return 0.0

    def asian_call(self,daily_returns):
        p=self.p0
        daily_prices=[]
        for r in daily_returns:
            p=p*exp(r)
            daily_prices.append(p)
        p=sum(daily_prices[-5:])/5
        return max(p-self.s,0)*exp(-self.r_free*self.days/250)

class OptionJumpDiffusion(OptionPricerSimple):

    def __init__(self,p0,mu,sigma,ndays,s,
                 xm=0,alpha=1,lamb=0,
                 r_free=0.02,option_type=None):
        self.p0=p0
        self.mu=mu
        self.sigma=sigma
        self.ndays=ndays
        self.s=s
        self.r_free=r_free
        self.option_type=option_type
        self.lamb=lamb
        self.xm=xm
        self.alpha=alpha

    def simulate_once(self):
        daily_returns=[]
        next_jump=random.expovariate(self.lamb)
        for t in range(self.ndays):
            if t==int(next_jump):
                r=-self.xm/random.random()**self.alpha #pareto
                next_jump=t+1+random.expovariate(self.lamb)
            else:
                r=random.gauss(self.mu,self.sigma)
            daily_returns.append(r)
        return self.option_type(daily_returns)


s=Stock('GOOG')
h=s.historical()
price_today = h[-1].close
log_returns=[row.log_return for row in h[-250:]]

r=OptionPricerSimple(p0=price_today,log_returns=log_returns,days=30,s=550)
r.option_type=r.european_call
#r.option_type=r.european_put
mu,dmu=r.simulate_many()
print mu, dmu
