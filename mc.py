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
    WITHOUT Gaussian assumptions

    """
    mu = mean(x)
    means=[]
    for k in range(nsamples):
        sample = get_sample(x,len(x))
        means.append(mean(sample))
    means.sort()
    left_prob=int(((100.0-prob)/200)*nsamples)
    return means[left_prob], mu, means[nsamples-1-left_prob]

class MCSimulator:

    """Monte Carlo Simulator parent class.

    Run a simulation many times, given a set of input parameters.
    (Must be extended to provide the simulation method that will
    be run many times.)

    """

    def __init__(self,params=None):
        """Set initial member variables.

        Keyword argument:
        paramsl -- Any input parameters needed for the
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

        Keyword arguments:
        absolute_precision -- precision (from 0 to 1) of the
            standard error of the mean that will end the
            simulation
        relative_precision -- precision, relative to the
            average of the simulation results, of the standard
            error of the mean that will end the simulation
        max_iterations -- maximum number of individual simulations
            to run before stopping

        """
        i=0
        mu=0.0
        var=0.0
        while True:
            y = self.simulate_once()
            self.results.append(y)
            mu = (mu*i+y)/(i+1) 
            var = (var*i+(y-mu)**2)/(i+1) 			
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
        return bootstrap(self.results)

    def limits(self,confidence=0.90):
        """Return the left and right limits.

        Sort all results and find the bounds where
            the percentage of results given in the
            confidence level are within the bounds.

        """
        self.results.sort()
        left_tail = (1.0-confidence)/2
        right_tail = 1.0-left_tail
        return self.results[int(left_tail*len(self.results))], \
               self.results[int(right_tail*len(self.results))]
