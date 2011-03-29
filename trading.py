# import mathematical module
import math
# import least squares fitting function
from numeric import fit, QUADRATIC
# import class to download Yahoo financial data
from yahoo import Stock

# we create a function to download adjusted closing prices from Yahoo
def download(symbol='aapl',days=360):
    return [d.adjusted_close for d in Stock(symbol).historical()[-days:]]

# we implement a forecast model
def model(window):
    n = len(window)
    # we fit last few days quadratically
    a,chi2,ff=fit(QUADRATIC,range(n),window)
    # and we extrapolate tomorrow's price
    price_tomorrow = ff(n)  
    return price_tomorrow

# we implement a trading strategy based on our model forecast
def sample_strategy(window):
    price_today = window[-1]
    price_tomorrow = model(window)
    if price_tomorrow>price_today:
        return 'buy'
    else:
        return 'sell'

# we run our strategy
def simulate(data,deposit=1000.0,shares=0.0,days=7,daily_rate=0.03/360,
             strategy=sample_strategy):
    # we make a sliding window
    for t in range(days,len(data)):
        window =  data[t-days:t]
        today_close = window[-1]
        suggestion = strategy(window)
        # and we buy or sell based on our strategy
        if deposit>0 and suggestion=='buy':
            # we keep track of finances
            shares += int(deposit/today_close)
            deposit -= shares*today_close
        elif shares>0 and suggestion=='sell':
            deposit += shares*today_close
            shares = 0.0
        # we assume money in the bank also gains an interest
        deposit*=math.exp(daily_rate)
        print t, suggestion, deposit, shares, deposit+shares*window[-1]        
    # we return the net worth
    return deposit+shares*data[-1]

# now we are ready to download the data: one year of closing princes for Apple
data = download('aapl',360)
# we run our strategy assuming we invested $1000 one year ago
# and we compute the net worth at the end of term
print simulate(data,deposit=1000.0)
# now we compare with the net worth if we invested the money at a fixed 3% annual interest
print 1000.0*math.exp(0.03)
