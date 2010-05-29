"""Monte Carlo Simulation classes and functions

    MCSimulator Class
    -----------------
        Class that runs our simulations many times. (Must be extended
        to provide the simulation method that will be run many times.)

    bootstrap Function
    ------------------
        Function that provides returns the bootstrap of a given
        sample with a given probability

    get_sample Function
    -------------------
        Function to return a sample of a given size. (Used by
        the bootstrap function.)

"""

from math import sqrt
import random

#Remove this function when we integrate with rest of code
def mean(sample):
    return sum(sample)/len(sample)

def get_sample(x,size):
    """Get a sample from the input list.

    Build a random subset of the input list with the given
    length.
    Parameters:
        x: List from which to pull a random sample
        size: Number of elements in the resulting sample
    Examples:
    >>> x=[1,2,3,4,5,6,7]
    >>> print get_sample(x,7)
    [3, 7, 5, 3, 5, 7, 4]
    >>> print get_sample(x,7)
    [1, 1, 5, 7, 7, 2, 3]
    >>> print get_sample(x,5)
    [7, 1, 4, 4, 1]
    >>> print get_sample(x,3)
    [7, 6, 4]
    
    """
    sample=[]
    for i in range(size):
        j = random.randint(0,len(x)-1)
        sample.append(x[j])
    return sample

def bootstrap(x,prob=68,nsamples=100):
    """Return the bootstrap of the input list.

    Take the input list, and return the mean and
    left and right standard error of the mean
    WITHOUT Gaussian assumptions (bootstrap)
    Parameters:
        x: List from which to create the bootstrap
        prob: Probability 
        nsamples: Number of sample means to obtain
    Example: (should use larger samples than this example)
    >>> y=[1,2,3,4,5,6,7,8,9,10]    
    >>> print bootstrap(y,68,10)
    (4, 5, 5)
    >>> print bootstrap(y,68,10)
    (3, 5, 6)
    >>> print bootstrap(y,68,10)
    (5, 5, 6)
    >>> print bootstrap(y,68,10)
    (4, 5, 6)
    
    """
    mu = mean(x)
    means=[]
    for k in range(nsamples):
        sample = get_sample(x,len(x))
        means.append(mean(sample))
    means.sort()
    left_prob=int(((100.0-prob)/200)*nsamples)
    return means[left_prob], mu, means[nsamples-1-left_prob]

def confidence_intervals(mu,sigma):
    """Return the set of confidence intervals, given an average and sigma

    Parameters:
        mu: Average of a gaussian distribution
        sigma: Square root of the variance of the distribution
    Examples:
    >>> confidence_intervals(10,1)
    0.68% between 9.0 and 11.0
    0.8% between 8.71844843446 and 11.2815515655
    0.9% between 8.35514637305 and 11.644853627
    0.95% between 8.04003601546 and 11.9599639845
    0.98% between 7.67365212596 and 12.326347874
    0.99% between 7.42417069645 and 12.5758293035
    0.995% between 7.19296623166 and 12.8070337683
    0.998% between 6.90976769383 and 13.0902323062
    0.999% between 6.70947326851 and 13.2905267315
    0.9999% between 6.10940811359 and 13.8905918864
    0.99999% between 5.58282658653 and 14.4171734135

    """
    CONFIDENCE=[
        (0.68,1.0),
        (0.80,1.281551565545),
        (0.90,1.644853626951),
        (0.95,1.959963984540),
        (0.98,2.326347874041),
        (0.99,2.575829303549),
        (0.995,2.807033768344),
        (0.998,3.090232306168),
        (0.999,3.290526731492),
        (0.9999,3.890591886413),
        (0.99999,4.417173413469)
        ]
    for (a,b) in CONFIDENCE:
        print '%s%% between %s and %s' % (a,mu-b*sigma,mu+b*sigma)

class MCSimulator:

    """Monte Carlo Simulator parent class.

    Run a simulation many times, given a set of input parameters.
    (Must be extended to provide the simulation method that will
    be run many times.)

    """

    def __init__(self,params=None):
        """Set initial member variables.

        Parameter:
            params: Any input parameters needed for the
                simulate_once method of the child class

        """
        self.params=params
        self.results=[]
				        
    def simulate_once(self):
        """Method that must be extended and will be run many times.

        Must contain the algorithm the we are using for our simulation.

        """
        return 0 

    def simulate_many(self,
                      absolute_precision=0.1,
                      relative_precision=0.1,
                      max_iterations=1000):
        """Run the simultate_once method many times.

        Parameters:
            absolute_precision: precision (from 0 to 1) of the
                standard error of the mean that will end the
                simulation
            relative_precision: precision, relative to the
                average of the simulation results, of the standard
                error of the mean that will end the simulation
            max_iterations: maximum number of individual simulations
                to run before stopping
        Example:
        (After extending MCSimulator class with a simulate_once()
         method)
        >>> r.simulate_many()
        (2.0058572140945952, 2.2502946031657665, 2.4938213919677454)
        >>> r.simulate_many()
        (2.0421917987142866, 2.2462842826554703, 2.5042897115800313)
        >>> r.simulate_many()
        (2.060113780332697, 2.2329663916515838, 2.4469048076040187)
        
        """
        i=0
        mu=0.0
        var=0.0
        while True:
            #Loop and run simulate_once() many times.
            y = self.simulate_once()
            #Collect results.
            self.results.append(y)
            #Maintain the average and variance of results
            mu = (mu*i+y)/(i+1) 
            var = (var*i+(y-mu)**2)/(i+1)
            #After simulating at least ten times, begin
            #    checking the precision for when to stop
            if i>10:
                sigma= sqrt(var)
                dmu=sigma/sqrt(i)
                if abs(dmu)<absolute_precision:
                    break
                if abs(dmu)<abs(mu)*relative_precision:
                    break
            i=i+1			
            if i>=max_iterations:
                break
            #bootstrap the results to get a non-gaussian
            #    average and standard error
        return bootstrap(self.results)
        """ Do we also need to return the confidence intervals? """

    def limits(self,confidence=0.90):
        """Return the left and right limits.

        Sort all results and find the bounds where the
            percentage of results given in the confidence
            level are within the returned bounds.
        examples:
        >>> r.limits()
        (0.0, 18.805436076518593)
        >>> r.limits(0.80)
        (0.0, 1.9799101201150022)
        >>> r.limits(0.90)
        (0.0, 18.805436076518593)
        >>> r.limits(0.95)
        (0.0, 30.674311239385283)
        >>> r.limits(0.99)
        (0.0, 59.512170255239283)
    
        """
        self.results.sort()
        left_tail = (1.0-confidence)/2
        right_tail = 1.0-left_tail
        return self.results[int(left_tail*len(self.results))], \
               self.results[int(right_tail*len(self.results))]
