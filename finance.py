from yahoo import *
from mc import *
from numeric import *
import random

def npv(amount,days,annual_risk_free_rate=0.05,days_in_year=250,probability=1.0):
       """
       compute the net presemt value of a financial transaction that pays amount after days with given probability
       assuming a fixed risk free rate and a fixed number of trading days in one year
       """
       return probability*amount*exp(-days*annual_risk_free_rate/days_in_year)

#exp(-self.r_free*self.days/250)
"""
this is the old formula we were using to computer present value
"""

class OptionPricerSimple(MCSimulator):

    def __init__(self,p0,log_returns,days,s,option_type=None):
        MCSimulator.__init__(self)
        self.p0=p0
        self.log_returns=log_returns
        self.days=days
        self.s=s
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
        put = max(self.s-p,0)
        return npv(call,self.days)

    def european_call(self,daily_returns):
        r=sum(daily_returns)
        p = exp(r)*self.p0
        call = max(p-self.s,0)
        return npv(call,self.days)

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
        asian_call = max(p-self.s,0)
        return npv(asian_call,self.days)

class OptionJumpDiffusion(OptionPricerSimple):

    def __init__(self,p0,mu,sigma,days,s,
                 xm=0,alpha=1,lamb=0,
                 r_free=0.02,option_type=None):
        OptionPricerSimple.__init__(self,p0,log_returns,days,s,option_type=None)
        self.p0=p0
        self.mu=mu
        self.sigma=sigma
        self.days=days
        self.s=s
        self.option_type=option_type
        self.lamb=lamb
        self.xm=xm
        self.alpha=alpha

    def simulate_once(self):
        daily_returns=[]
        next_jump=random.expovariate(self.lamb)
        for t in range(self.days):
            if t==int(next_jump):
                r=-self.xm/random.random()**self.alpha #pareto
                next_jump=t+1+random.expovariate(self.lamb)
            else:
                r=random.gauss(self.mu,self.sigma)
            daily_returns.append(r)
        return self.option_type(daily_returns)

class Market(MCSimulator):
    def __init__(self,days,positions={}):
        MCSimulator.__init__(self)
        self.days=days
        self.data=data={}
        for name in positions:
            shares=positions[name]
            data[name]={}
            stock=Stock(name)
            data[name]['shares']=shares
            data[name]['stock']=stock
            data[name]['current_price']=float(stock.current().price)
            historical=stock.historical()
            data[name]['log_returns'] = \
                [day.log_return for day in historical[-250:]]

    def simulate_once(self):
        shares=1.0
        #days=10        

        S={}
        for name in self.data:
            S[name]=self.data[name]['current_price']
        n=250
        for t in range(self.days):
            i=random.randint(0,n-1)
            for name in self.data:
                r=self.data[name]['log_returns'][i]
                S[name]=exp(r)*S[name]
        # return total portfolio value
        return sum(self.data[name]['shares']*S[name] for name in self.data)

    def value_at_risk(self):
       mu = mean(self.results)
       sigma = stddev(self.results)
       confidence_intervals(mu,sigma)
       var_gaussian = mu - 1.65 * sigma
       n=len(self.results)
       var_nongaussian=self.results[int(n*5/100)]
       print var_gaussian, var_nongaussian
       return   self.limits(confidence=0.95)[0]

class OpRiskModel(MCSimulator):

    def __init__(self,lamb,xm,alpha,days):
        MCSimulator.__init__(self)
        self.lamb=lamb
        self.xm=xm
        self.alpha=alpha
        self.days=days

    def simulate_once(self):
        t=random.expovariate(self.lamb)
        loss=0.0
        while t<self.days:
            t+=random.expovariate(self.lamb)
            amount=self.xm*random.paretovariate(self.alpha)
            loss+=amount
        return loss

#Option Pricer Simpler
s=Stock('GOOG')
h=s.historical()
price_today = h[-1].close
log_returns=[row.log_return for row in h[-250:]]

r=OptionPricerSimple(p0=price_today,log_returns=log_returns,days=30,s=450)
r.option_type=r.european_call
#r.option_type=r.european_put
lower,mu,upper=r.simulate_many()
print lower, mu, upper

#Option Jump Diffuser
s=Stock('AAPL')
h=s.historical()
price_today = h[-1].close

jump=-0.05
log_returns=[row.log_return for row in h[-250:]]
mu = E(lambda r:r, [r for r in log_returns if r>jump])
variance = E(lambda r: (r-mu)**2,[r for r in log_returns if r>jump])
sigma=sqrt(variance)
jumps=[r for r in log_returns if r<=jump]
lamb=float(len(jumps))/250
xm=0.05
average_jump=-sum(jumps)/len(jumps)
alpha=average_jump/(average_jump-xm)

print 'mu=',mu
print 'sigma=',sigma
print 'lamb=',lamb
print 'xm=',xm
print 'alpha=',alpha

r=OptionJumpDiffusion(p0=price_today,mu=mu,sigma=sigma,days=30,s=250,xm=xm,alpha=alpha,lamb=lamb)
r.option_type=r.european_call
#r.option_type=r.european_put
lower,mu,upper=r.simulate_many()
print lower,mu,upper


#Market Portfolio
days = 25
positions={'XOM':7,'AAPL':1}
M=Market(days,positions)
lower, mu, upper = M.simulate_many()
print M.value_at_risk()
print lower,mu,upper

#Operation Risk Model

r=OpRiskModel(lamb=15.22,xm=5000,alpha=1.208,days=365)
lower, mu, upper = r.simulate_many()
print lower,mu,upper
