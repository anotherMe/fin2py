#Initial Add...still need to add comments and figure out how to handle "params"
from math import sqrt
import random

#Remove once we integrate with rest of code
def mean(sample):
    return sum(sample)/len(sample)

def get_sample(x,size):
    sample=[]
    for i in range(size):
        j = random.randint(0,len(x)-1)
        sample.append(x[j])
    return sample

def bootstrap(x,prob=68,nsamples=100):
    mu = mean(x)
    means=[]
    for k in range(nsamples):
        sample = get_sample(x,len(x))
        means.append(mean(sample))
    means.sort()
    left_prob=int(((100.0-prob)/200)*nsamples)
    return means[left_prob], mu, means[nsamples-1-left_prob]

class MCSimulator:

    def __init__(self,params=None):
        self.params=params
        self.results=[]
				        
    def simulate_once(self):
        """
        comments
        """
        return 0 

    def simulate_many(self,
                      absolute_precision=0.1,
                      relative_precision=0.1,
                      max_iterations=1000):
        """
        comments
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
        print "i=",i      
        return bootstrap(self.results)

    def limits(self,confidence=0.90):
        """
        comments
        """
        self.results.sort()
        left_tail = (1.0-confidence)/2
        right_tail = 1.0-left_tail
        return self.results[int(left_tail*len(self.results))], \
               self.results[int(right_tail*len(self.results))]
