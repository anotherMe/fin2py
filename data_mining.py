# Constants defining The neuron's response curve
# from http://code.activestate.com/recipes/496908-data-mining-with-neural-nets/
# Created by Created by Raymond Hettinger on Wed, 26 Jul 2006 (PSF) 
# modified by Massimo Di Pierro
# License PSF

__all__ = 'NeuralBusinessIntelligence'

minact, rest, thresh, decay, maxact = -0.2, -0.1, 0.0, 0.1, 1.0
alpha, gamma, estr = 0.1, 0.1, 0.4

class Unit(object):
    __slots__ = ['name', 'pool', 'unitbyname', 'extinp', 'activation', 'output', 'exciters', 'newact']
    def __init__(self, name, pool, unitbyname):
        self.name = name
        self.pool = pool
        self.unitbyname = unitbyname
        self.reset()
        self.exciters = []
        self.unitbyname[name] = self
    def reset(self):
        self.setext(0.0)
        self._setactivation()
    def setext(self, weight=1.0):
        self.extinp = weight
    def _setactivation(self, val=rest):
        self.activation = val
        self.output = max(thresh, val)
    def addexciter(self, aunit):
        self.exciters.append(aunit)
    def remove(self, aunit):
        self.exciters.remove(aunit)
    def computenewact(self):
        ai = self.activation
        plus = sum(exciter.output for exciter in self.exciters)
        minus = self.pool.sum - self.output
        netinput = alpha*plus - gamma*minus + estr*self.extinp
        if netinput > 0:
            ai = (maxact-ai)*netinput - decay*(ai-rest) + ai
        else:
            ai = (ai-minact)*netinput - decay*(ai-rest) + ai
        self.newact = max(min(ai, maxact), minact)
    def commitnewact(self):
        self._setactivation(self.newact)

class Pool(object):
    __slots__ = ['sum', 'members','unitbyname']
    def __init__(self, unitbyname):
        self.sum = 0.0
        self.members = set()
        self.unitbyname = unitbyname
    def addmember(self, member):
        self.members.add(member)
    def updatesum(self):
        self.sum = sum(member.output for member in self.members)
    def display(self):
        result = sorted(((unit.activation, unit.name) for unit in self.members),
                        reverse=True)
        for i, (act, self.unitbyname) in enumerate(result):
            print '%s: %.2f' % (self.unitbyname, act)
        print '\n'



class NeuralBusinessIntelligence:
    __slots__ = ['units', 'pools', 'unitbyname']
    def __init__(self):
        self.units = []
        self.pools = []
        self.unitbyname = {}

    def load_filename(self, filename):
        """Load in a database and interpret it as a network
        
        First column must be unique keys which define the instance units.
        Each column is a pool (names, gangs, ages, etc).
        Every row is mutually excitory.
        """
        self.units[:] = []
        self.pools[:] = []
        for line in open(filename):
            relatedunits = line.split()
            if not len(relatedunits): continue
            key = len(self.units)
            for poolnum, name in enumerate(relatedunits):
                if poolnum >= len(self.pools):
                    self.pools.append(Pool(self.unitbyname))
                pool = self.pools[poolnum]
                if name in self.unitbyname:
                    unit = self.unitbyname[name]
                else:
                    unit = Unit(name, pool, self.unitbyname)
                    self.units.append(unit)
                pool.addmember(unit)
                if poolnum > 0:
                    self.units[key].addexciter(unit)
                    unit.addexciter(current)

    def load_records(self, records):
        """Load in a database and interpret it as a network
        
        First column must be unique keys which define the instance units.
        Each column is a pool (names, gangs, ages, etc).
        Every row is mutually excitory.
        """
        self.units[:] = []
        self.pools[:] = []
        for counter,record in enumerate(records):
            relatedunits = ['%s:%s' % (k,v) for (k,v) in record.items()]
            relatedunits.insert(0,'_:%s'% counter)
            if not len(relatedunits): continue
            key = len(self.units)
            for poolnum, name in enumerate(relatedunits):
                if poolnum >= len(self.pools):
                    self.pools.append(Pool(self.unitbyname))
                pool = self.pools[poolnum]
                if name in self.unitbyname:
                    unit = self.unitbyname[name]
                else:
                    unit = Unit(name, pool, self.unitbyname)
                    self.units.append(unit)
                pool.addmember(unit)
                if poolnum > 0:
                    self.units[key].addexciter(unit)
                    unit.addexciter(self.units[key])

    def reset(self):
        for unit in self.units:
            unit.reset()

    def depair(self,i, j):
        self.unitbyname[i].remove(self.unitbyname[j])
        self.unitbyname[j].remove(self.unitbyname[i])

    def touch(self, name, weight=1.0):
        self.unitbyname[name].setext(weight)

    def run(self,times=1000):
        """Run n-cycles and display result"""
        for i in xrange(times):
            for pool in self.pools:
                pool.updatesum()
            for unit in self.units:
                unit.computenewact()
            for unit in self.units:
                unit.commitnewact()
        print '-' * 20
        for pool in self.pools:
            pool.display()

SAMPLE_DATA = """
Art         Jets        40      jh      sing    pusher
Al          Jets        30      jh      mar     burglar
Sam         Jets        20      col     sing    bookie
Clyde       Jets        40      jh      sing    bookie
Mike        Jets        30      jh      sing    bookie
Jim         Jets        20      jh      div     burglar
Greg        Jets        20      hs      mar     pusher
John        Jets        20      jh      mar     burglar
Doug        Jets        30      hs      sing    bookie
Lance       Jets        20      jh      mar     burglar
George      Jets        20      jh      div     burglar
Pete        Jets        20      hs      sing    bookie
Fred        Jets        20      hs      sing    pusher
Gene        Jets        20      col     sing    pusher
Ralph       Jets        30      jh      sing    pusher

Phil        Sharks      30      col     mar     pusher
Ike         Sharks      30      jh      sing    bookie
Nick        Sharks      30      hs      sing    pusher
Don         Sharks      30      col     mar     burglar
Ned         Sharks      30      col     mar     bookie
Karl        Sharks      40      hs      mar     bookie
Ken         Sharks      20      hs      sing    burglar
Earl        Sharks      40      hs      mar     burglar
Rick        Sharks      30      hs      div     burglar
Ol          Sharks      30      col     mar     pusher
Neal        Sharks      30      hs      sing    bookie
Dave        Sharks      30      hs      div     pusher
"""

def test1():
    open('jets.txt','w').write(SAMPLE_DATA)
    bi = NeuralBusinessIntelligence()
    bi.load_filename('jets.txt')
    bi.touch('Ken', weight=0.8)
    bi.touch('20', weight=0.8)
    bi.run()

    bi.reset()
    for key in ('Sharks','20','jh','sing','burglar'):
        bi.touch(key)
    bi.run()

    bi.reset()
    bi.touch('Lance')
    bi.depair('Lance','burglar')
    bi.run()

def test2map(items):
    return {'name':items[0],
            'team':items[1],
            'age':items[2],
            'type':items[3],
            'hobby':items[4],
            'activity':items[5]}

def test2():
    records = [test2map(item.split()) for item in SAMPLE_DATA.split('\n') if item]
    bi = NeuralBusinessIntelligence()
    bi.load_records(records)
    bi.touch('name:Ken',weight=0.8)
    bi.touch('age:20',weight=0.8)
    bi.run()

def test_db():
    """
    rows:  ### consider all people who have dogs
      db(db.dog.owner==db.person.id).select()
    input: ### and people meeting these conditions
      db.person.age>20
      db.person.name.like('C%')
    out:   ### what is the probability that    
      row.dog.name    # dog names
      row.dog.age>5   # and the dog is older than 5
    """
    from gluon.contrib.populate import populate
    db=DAL()
    db.define_table('person',Field('name'),Field('age','integer',requires=IS_INT_IN_RANGE(1,80)))
    db.define_table('dog',Field('name'),Field('owner',db.person),Field('age','integer',requires=IS_INT_IN_RANGE(1,20)))
    populate(db.person,20)
    db.commit()
    populate(db.dog,20)
    db.commit()
    
    rows=db(db.dog.owner==db.person.id).select()
    input = {
        'person mature':True,
        'person name with C':True
        }
    output =  {
        'person mature':(lambda row: row.person.age>10),
        'person name with C':(lambda row: row.person.name.startswith('C')),
        'dog name':(lambda row: row.dog.name),
        'dog old':(lambda row: row.dog.age>5)
        }
    records=[]
    for row in rows:
        d={}
        records.append(d)
        for key,f in output.items():
            d[key]=f(row)
            print '%s:%s\t' % (key,d[key]),
        print

    bi = NeuralBusinessIntelligence()
    bi.load_records(records)
    for key,value in input.items():
        try:
            bi.touch("%s:%s" % (key,value), weight=0.8)
        except:
            print 'no data for %s:%s' % (key,value)
    bi.run()    
            
if __name__ == '__main__' or 1:
    test_db()

