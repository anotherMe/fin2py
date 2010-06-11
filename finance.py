"""
finance.py

       Functions
       ----------
       npv

       Classes
       -------
       Params
       OptionPricerSimple
       OptionJumpDiffusion
       Market
       OpRiskModel
             
"""

from yahoo import *
from mc import *
from math import *
from numeric import *
import random
from copy import deepcopy

__all__=['npv','Markowitz',
         'OptionPricerSimple','OptionJumpDiffusion','Market','OpRiskModel']

def npv(amount,days,annual_risk_free_rate=0.05,days_in_year=250):
    """Compute the NPV of certain amount using fixed rate.
         
    Compute the net presemt value of a financial transaction that pays
    amount after days with given probability assuming a fixed risk free
    rate and a fixed number of trading days in one year.

    Parameter
    ----------
    amount : float
        The amount to be discounted to present value.
    days : float
        Number of days for the present time.
    annual_risk_free_rate : float
        The risk free rate, or the discount rate. Defaulted to be 0.05.
    days_in_year : int
        Number of trading days in a year. Default is 250.

    Return
    ------
    npv : float
          Net present value.

    Example:
    >>> print npv(amount=550.75,days=180,annual_risk_free_rate=0.02,days_in_year=365)
    545.344645567
    """
    return amount*exp(-days*annual_risk_free_rate/days_in_year)

class Markowitz:
    def __init__(self,returns, cov, r_free=0.05, checkpoint=None):
        self.returns = returns
        self.returnsv=[[r] for r in self.returns]
        self.cov=cov
        self.inv_cov=inverse(cov,checkpoint)
        self.r_free=r_free
        
        
    def normalize(self,v):
        v=deepcopy(v)
        norm=0.0
        for x in v: norm+=x[0]
        for x in v: x[0]=float(x[0])/abs(norm)
        return v
    
    def risk_return(self,x_mar):
        n=len(self.returnsv)
        r_freev=[[self.r_free]]*n
        x_free=1.0-sum([x[0] for x in x_mar])
        r=multiply(transpose(self.returnsv),x_mar)[0][0]+x_free*self.r_free
        sigma=sqrt(multiply(transpose(x_mar),multiply(self.cov,x_mar))[0][0])
        return x_free, r, sigma

    def optimize(self):    
        n=len(self.returns)    
        r_freev=[[self.r_free]]*n
        ones=[[1.0]]*n
        invSI=multiply(self.inv_cov,ones)
        IinvSI=multiply(transpose(ones),invSI)[0][0]
        invSr=multiply(self.inv_cov,self.returnsv)   
        IinvSr=multiply(transpose(ones),invSr)[0][0]
        rinvSr=multiply(transpose(self.returnsv),invSr)[0][0]
        # portfolio min variance
        x_min=self.normalize(invSI)
        r_min=multiply(transpose(self.returnsv),x_min)[0][0]
        sigma_min=sqrt(multiply(transpose(x_min),multiply(self.cov,x_min))[0][0])
        # portfolio Markowitz
        invSr=multiply(self.inv_cov,sub(self.returnsv,r_freev))
        x_mar=self.normalize(invSr)
        x_free,r_mar,sigma_mar=self.risk_return(x_mar)
        x_mar=[x[0] for x in x_mar]
        # parameters of hyperbola
        a2=1.0/IinvSI
        b2=(rinvSr*IinvSI-IinvSr**2)*a2*a2
        # hyperbola sigma^2/a2 - (r-r_min)^2/b2=1
        return x_mar, r_mar, sigma_mar, (r_min, a2, b2)

def test_Markowitz():
    returns = [random.random()/10 for i in range(10)]
    cov = [[ (1.0 if i==j else 0)+0.01*(i+j)*(i-j)**2 for i in range(10)] for j in range(10)]
    m=Markowitz(returns,cov)
    print m.optimize()


class OptionPricerSimple(MCSimulator):
    """Simple option pricer

    Price options provided different option types.
    
    """

    def __init__(self,params,option_type=None):
        """Initialize the pricer.

        Parameter
        ---------
        params : object
            Contains all kinds of information needed for an option (simulation).

        Example:

        s=Stock('GOOG')
        h=s.historical()
        price_today = h[-1].close
        log_returns=[row.log_return for row in h[-250:]]
        params=Storage()
        params.p0=price_today
        params.log_returns=log_returns
        params.days=30
        params.s=450
        r=OptionPricerSimple(params)
        """
        MCSimulator.__init__(self)
        self.params=params
        self.option_type=option_type

    def simulate_once(self):
        """Simulate once

        Run the simulation once and return the NPV of the option provided
        different types of options.
       
        Return
        ------
        Option price : float
            The option price, given a option type.

        Example:
        
        r=OptionPricerSimple(params)
        """
        daily_returns=[]
        for t in range(self.params.days):
            i=random.randint(0,len(self.params.log_returns)-1)
            daily_returns.append(self.params.log_returns[i])
        return self.option_type(daily_returns)

    def european_put(self,daily_returns):
        """A european put option
       
        Parameter
        ---------
        daily_returns : list
            One-dimensional number list, containing simulated log returns
            before the option's expiration.

        Return
        ------
        npv : float
            The NPV of the option price.


        Example:

        r.option_type=r.european_put
        lower,mu,upper=r.simulate_many()
        print "Bootstrap: ",lower, mu, upper

        Output
        0.68% between -4.80943057248 and 6.78911186586
        0.8% between -6.44222446326 and 8.42190575664
        0.9% between -8.54911165184 and 10.5287929452
        0.95% between -10.3765220795 and 12.3562033728
        0.98% between -12.501281625 and 14.4809629184
        0.99% between -13.9480920989 and 15.9277733923
        0.995% between -15.2889094973 and 17.2685907907
        0.998% between -16.931254627 and 18.9109359204
        0.999% between -18.0928163232 and 20.0724976166
        0.9999% between -21.5727569057 and 23.5524381991
        0.99999% between -24.6265460001 and 26.6062272935
        Bootstrap:  0.836162945606 0.989840646689 1.21639993093
        """
        r=sum(daily_returns)
        p = exp(r)*self.params.p0
        put = max(self.params.s-p,0)
        return npv(put,self.params.days)

    def european_call(self,daily_returns):
        """A european call option
       
        Parameter
        ---------
        daily_returns : list
            One-dimensional number list, containing simulated log returns
            before the option's expiration.

        Return
        ------
        npv : float
            The NPV of the option price.

        Example:

        r.option_type=r.european_call
        lower,mu,upper=r.simulate_many()
        print "Bootstrap: ",lower, mu, upper

        Output
        0.68% between 41.5414432039 and 74.3782854259
        0.8% between 36.9188110364 and 79.0009175934
        0.9% between 30.9539648017 and 84.9657638281
        0.95% between 25.7803502544 and 90.1393783754
        0.98% between 19.7649052683 and 96.1548233615
        0.99% between 15.6688140993 and 100.250914531
        0.995% between 11.8728018335 and 104.046926796
        0.998% between 7.22312898155 and 108.696599648
        0.999% between 3.9346107604 and 111.985117869
        0.9999% between -5.9175116471 and 121.837240277
        0.99999% between -14.5631489076 and 130.482877537
        Bootstrap:  52.6002843997 57.9598643149 62.9774751068
        """
        r=sum(daily_returns)
        p = exp(r)*self.params.p0
        call = max(p-self.params.s,0)
        return npv(call,self.params.days)

    def digital_put(self,daily_returns):
        """A digital put option

        Return 1 if the strike price exceeds the actual price.
       
        Parameter
        ---------
        daily_returns : list
            One-dimensional number list, containing simulated log returns
            before the option's expiration.

        Return
        ------
        digital_put : float
            1.0 if the strike price exceeds the actual price, 0 otherwise.

        Example:

        r.option_type=r.digital_put
        lower,mu,upper=r.simulate_many()
        print "Bootstrap: ",lower, mu, upper

        Output
        0.68% between -0.182546572416 and 0.490238880109
        0.8% between -0.277258471134 and 0.584950778826
        0.9% between -0.399470642027 and 0.707162949719
        0.95% between -0.50547147429 and 0.813163781982
        0.98% between -0.628720349738 and 0.93641265743
        0.99% between -0.712644087962 and 1.02033639565
        0.995% between -0.790419588198 and 1.09811189589
        0.998% between -0.88568551641 and 1.1933778241
        0.999% between -0.9530631042 and 1.26075541189
        0.9999% between -1.1549206576 and 1.46261296529
        0.99999% between -1.33205885309 and 1.63975116078
        Bootstrap:  0.0769230769231 0.153846153846 0.230769230769
        """
        r=sum(daily_returns)
        p = exp(r)*self.params.p0
        if self.params.s>p: return 1.0
        return 0.0

    def digital_call(self,daily_returns):
        """A digital call option

        Return 1 if the strike price is lower than the actual price.
       
        Parameter
        ---------
        daily_returns : list
            One-dimensional number list, containing simulated log returns
            before the option's expiration.

        Return
        ------
        digital_call : float
            1.0 if the strike price is lower than the actual price, 0 otherwise.

        Example:
        
        r.option_type=r.digital_call
        lower,mu,upper=r.simulate_many()
        print "Bootstrap: ",lower, mu, upper 

        Output:
        0.68% between 0.65600527008 and 1.17732806325
        0.8% between 0.582615645794 and 1.25071768754
        0.9% between 0.487916823085 and 1.34541651025
        0.95% between 0.405779717197 and 1.42755361614
        0.98% between 0.310277580873 and 1.52305575246
        0.99% between 0.245247403035 and 1.5880859303
        0.995% between 0.184981324345 and 1.64835200899
        0.998% between 0.111162397964 and 1.72217093537
        0.999% between 0.0589533733311 and 1.77437996
        0.9999% between -0.0974604479935 and 1.93079378133
        0.99999% between -0.234719924252 and 2.06805325759
        Bootstrap:  0.833333333333 0.916666666667 1.0
        """        
        r=sum(daily_returns)
        p = exp(r)*self.params.p0
        if p>self.params.s:
            return 1.0
        return 0.0

    def asian_call(self,daily_returns):
        """A asian call option

        Return NPV of the option.
       
        Parameter
        ---------
        daily_returns : list
            One-dimensional number list, containing simulated log returns
            before the option's expiration.

        Return
        ------
        npv : float
            Return the NPV of the option price.

        Example:
        r.option_type=r.asian_call
        lower,mu,upper=r.simulate_many()
        print "Bootstrap: ",lower, mu, upper

        Output
        0.68% between 26.2187369775 and 70.6850627807
        0.8% between 19.9589551556 and 76.9448446027
        0.9% between 11.8816012419 and 85.0221985164
        0.95% between 4.8757013296 and 92.0280984286
        0.98% between -3.27017137019 and 100.173971128
        0.99% between -8.81693263336 and 105.720732392
        0.995% between -13.9573391627 and 110.861138921
        0.998% between -20.2537383877 and 117.157538146
        0.999% between -24.7069169742 and 121.610716732
        0.9999% between -38.0482633151 and 134.952063073
        0.99999% between -49.7558361871 and 146.659635945
        Bootstrap:  44.9457829948 48.4518998791 52.75656641
        """    
        p=self.params.p0
        daily_prices=[]
        for r in daily_returns:
            p=p*exp(r)
            daily_prices.append(p)
        p=sum(daily_prices[-5:])/5
        asian_call = max(p-self.params.s,0)
        return npv(asian_call,self.params.days)


class OptionJumpDiffusion(OptionPricerSimple):
    """Price options with jump diffusion.

    """

    def __init__(self,params,option_type=None):
        """Initialize the pricer.

        Parameter
        ---------
        params : object
            Contains all kinds of information needed for an option (simulation).

        Example:

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
        params=Storage()
        params.p0=price_today
        params.mu=mu
        params.sigma=sigma
        params.days=30
        params.s=250
        params.xm=xm
        params.alpha=alpha
        params.lamb=lamb        
        """
        OptionPricerSimple.__init__(self,params,option_type=None)
        self.params=params
        self.option_type=option_type

    def simulate_once(self):
        """Simulate once

        Run the simulation once and return the NPV of the option provided
        different types of options. The only diffence with that of the
        OptionPricerSimple is that the returns here involve jump-diffusion,
        and the returns incur jumps(pareto dist) which are of poisson process.
        Otherwise the returns follow a gaussian dist.
       
        Return
        ------
        Option price : float
            The option price, given a option type.

        Example:
        r=OptionJumpDiffusion(params)
        r.option_type=r.european_call
        lower,mu,upper=r.simulate_many()
        print "Bootstrap: ",lower,mu,upper

        Output:
        0.68% between 3.58388224084 and 48.8764799603
        0.8% between -2.79221865691 and 55.252580858
        0.9% between -11.0196657159 and 63.480027917
        0.95% between -18.1557490476 and 70.6161112487
        0.98% between -26.4529881066 and 78.9133503077
        0.99% between -32.1028191192 and 84.5631813203
        0.995% between -37.3387445267 and 89.7991067278
        0.998% between -43.7521432509 and 96.212505452
        0.999% between -48.2880706667 and 100.748432868
        0.9999% between -61.8773255003 and 114.337687701
        0.99999% between -73.802448136 and 126.262810337
        Bootstrap:  23.2061858388 26.2301811006 29.5317981192
        """        
        daily_returns=[]
        next_jump=random.expovariate(self.params.lamb)
        for t in range(self.params.days):
            if t==int(next_jump):
                r=-self.params.xm/random.random()**self.params.alpha #pareto
                next_jump=t+1+random.expovariate(self.params.lamb)
            else:
                r=random.gauss(self.params.mu,self.params.sigma)
            daily_returns.append(r)
        return self.option_type(daily_returns)

class Market(MCSimulator):
    """Compute the portfolio market value.

    """
    def __init__(self,params):
        """Initialize the portfolio.

        params
        ---------
        days : int
            Number of days to be simulated for the portfolio.
        positions : dict
            A portfolio dictionary containing stock names and the corresponding
            number of shares.

        Example:

        params=Storage()
        params.days = 25
        params.positions={'XOM':7,'AAPL':1}
        M=Market(params)
        """
        MCSimulator.__init__(self)
        self.params=params
        self.data=data={}
        for name in params.positions:
            shares=params.positions[name]
            data[name]={}
            stock=Stock(name)
            data[name]['shares']=shares
            data[name]['stock']=stock
            data[name]['current_price']=float(stock.current().price)
            historical=stock.historical()
            data[name]['log_returns'] = \
                [day.log_return for day in historical[-250:]]

    def simulate_once(self):
        """Simulate once

        Run the simulation once and return the market value of the portfolio.
        Simulated returns are pulled randomly from the actual returns.
       
        Return
        ------
        portfolio market value : float
            Sum of stocks in the portfolio time their simulated prices.

        Example:

        M=Market(params)
        lower, mu, upper = M.simulate_many()
        print "Bootstrap: ",lower,mu,upper

        Output:
        0.68% between 637.028749193 and 729.201229488
        0.8% between 624.053096129 and 742.176882551
        0.9% between 607.309870081 and 758.920108599
        0.95% between 592.787618468 and 773.442360212
        0.98% between 575.902362551 and 790.32761613
        0.99% between 564.404701478 and 801.825277202
        0.995% between 553.74935699 and 812.48062169
        0.998% between 540.697801167 and 825.532177513
        0.999% between 531.466984181 and 834.762994499
        0.9999% between 503.812237347 and 862.417741333
        0.99999% between 479.544074634 and 886.685904046
        Bootstrap:  665.941168272 683.11498934 700.574958866
        """  
        shares=1.0
        #days=10        

        S={}
        for name in self.data:
            S[name]=self.data[name]['current_price']
        n=250
        for t in range(self.params.days):
            i=random.randint(0,n-1)
            for name in self.data:
                r=self.data[name]['log_returns'][i]
                S[name]=exp(r)*S[name]
        # return total portfolio value
        return sum(self.data[name]['shares']*S[name] for name in self.data)

    def value_at_risk(self):
        """Compute the VAR of some stock.

        Return
        ------
        VAR : float

        Example:

        var = M.value_at_risk()
        print "VAR: ",var

        Output:
        Gaussian:  601.494337375 Non-Gaussian:  624.908115863
        VAR:  624.908115863
        """
        mu = mean(self.results)
        sigma = stddev(self.results)
        var_gaussian = mu - 1.65 * sigma
        n=len(self.results)
        self.results.sort()
        var_nongaussian=self.results[int(n*5/100)]
        print "Gaussian: ",var_gaussian, "Non-Gaussian: ",var_nongaussian
        return   self.limits(confidence=0.95)[0]

class OpRiskModel(MCSimulator):
    """Compute the total amount of losses
    """
    def __init__(self,params):
        """Initialize the risk model.

        Parameter
        ---------
        params : object
            Contains all kinds of information needed for risk model.

        Example:
        stor4=Storage()
        stor4.lamb=15.22
        stor4.xm=5000
        stor4.alpha=1.208
        stor4.days=365
        r=OpRiskModel(stor4)
        """
        MCSimulator.__init__(self)
        self.params=params

    def simulate_once(self):
        """Simulate once.

        Return
        ------
            This will return the amount of operational loss simulated for the
            given time period.

        Example:
        r=OpRiskModel(stor4)
        lower, mu, upper = r.simulate_many()
        print "Bootstrap: ",lower,mu,upper

        Output:
        0.68% between 127760271.155 and 162467836.895
        0.8% between 122874286.419 and 167353821.63
        0.9% between 116569621.33 and 173658486.72
        0.95% between 111101264.604 and 179126843.445
        0.98% between 104743118.138 and 185484989.911
        0.99% between 100413671.581 and 189814436.469
        0.995% between 96401399.4999 and 193826708.549
        0.998% between 91486833.5654 and 198741274.484
        0.999% between 88010967.5982 and 202217140.451
        0.9999% between 77597567.1919 and 212630540.857
        0.99999% between 68459385.7079 and 221768722.341
        Bootstrap:  138714608.714 145114054.025 150873917.501 
        """
        t=random.expovariate(self.params.lamb)
        loss=0.0
        while t<self.params.days:
            t+=random.expovariate(self.params.lamb)
            amount=self.params.xm*random.paretovariate(self.params.alpha)
            loss+=amount
        return loss


if __name__ == '__main__':
    import doctest
    test_Markowitz()
    doctest.testmod()
