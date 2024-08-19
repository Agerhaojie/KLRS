####################################################
#
# loss = reward*min(demand, order)-cost*order
#
###################################################

import coptpy as cp
from coptpy import COPT
import numpy as np
import math

def KL(p, q):
    distance = 0
    for i in range(1, len(p)):
        distance += p[i]*np.log(p[i]/q[i])
    return distance

class LOSS:
    def __init__(self, demand, reward, cost, order):
        self.num = len(demand)-1
        self.losses = [0]
        self.N1= range(1, self.num+1)
        self.lossmean = 0
        for n in self.N1:
            self.losses.append(reward*min(demand[n], order)-cost*order)
            self.lossmean+=self.losses[n]
        self.lossmean = self.lossmean/self.num


class ERM:
    def __init__(self, demand, reward, cost):
        self.samplenum = len(demand)-1
        self.demand = demand
        self.reward = reward
        self.cost = cost
        env = cp.Envr()

        self.ermmodel = env.createModel('ERMModel')
        self.ermmodel.setParam(COPT.Param.Logging, 0)
        self.ermvorder = self.ermmodel.addVar(lb=0, vtype=COPT.CONTINUOUS, name='vorder')

        N1 = range(1, self.samplenum + 1)
        self.ermvloss = [0]
        for n in N1:
            self.ermvloss.append(
                self.ermmodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vloss[{}]'.format(n)))

        self.ermvmin = [0]

        for n in N1:
            self.ermvmin.append(
                self.ermmodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vmin[{}]'.format(n)))

        self.ermmodel.addConstrs(self.ermvmin[n] <= self.ermvorder for n in N1)
        self.ermmodel.addConstrs(self.ermvmin[n] <= self.demand[n] for n in N1)

        self.ermcloss = [0]
        for n in N1:
            self.ermcloss.append(
                self.ermmodel.addConstr(self.reward * self.ermvmin[n] - cost * self.ermvorder == self.ermvloss[n], name='closs[{}]'.format(n)))

        ermobj = cp.quicksum(self.ermvloss[n] for n in N1)

        self.ermmodel.setObjective(ermobj / self.samplenum, sense=COPT.MAXIMIZE)

    def optimize(self):
        self.ermmodel.solve()
        
class KLRS:
    def __init__(self, demand, target, reward, cost):

        self.demand = demand
        self.reward = reward
        self.cost = cost
        self.samplenum = len(demand)-1
        env = cp.Envr()

        self.klrsmodel = env.createModel('KLRS')

        self.klrsmodel.setParam(COPT.Param.Logging, 0)
        self.klrsvorder = self.klrsmodel.addVar(lb=0, vtype=COPT.CONTINUOUS, name='vorder')
        self.klrsvlambda = self.klrsmodel.addVar(lb=0.001, vtype=COPT.CONTINUOUS, name='vlambda')

        self.N1 = range(1, self.samplenum + 1)
        self.klrsvloss = [0]
        for n in self.N1:
            self.klrsvloss.append(
                self.klrsmodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vloss[{}]'.format(n)))

        self.klrsvmin = [0]

        for n in self.N1:
            self.klrsvmin.append(
                self.klrsmodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vmin[{}]'.format(n)))

        self.klrsmodel.addConstrs(self.klrsvmin[n] <= self.klrsvorder for n in self.N1)
        self.klrsmodel.addConstrs(self.klrsvmin[n] <= self.demand[n] for n in self.N1)

        self.klrscloss = [0]
        for n in self.N1:
            self.klrscloss.append(
                self.klrsmodel.addConstr(self.reward * self.klrsvmin[n] - cost * self.klrsvorder == self.klrsvloss[n], name='closs[{}]'.format(n)))

        self.klrsvu = [0]
        for n in self.N1:
            self.klrsvu.append(self.klrsmodel.addVar(lb=-COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vu[{}]'.format(n)))

        self.klrsvexp1 = [0]

        for n in self.N1:
            self.klrsvexp1.append(self.klrsmodel.addVar(lb=0, vtype=COPT.CONTINUOUS, name='exploss[{}]'.format(n)))

        for n in self.N1:
            self.klrsmodel.addConstr(self.klrsvu[n] == -self.klrsvloss[n] + target)

        for n in self.N1:
            self.klrsmodel.addExpCone([self.klrsvexp1[n], self.klrsvlambda, self.klrsvu[n]], COPT.EXPCONE_PRIMAL)

        self.klrsmodel.addConstr(cp.quicksum(self.klrsvexp1[n] / self.samplenum for n in self.N1) == self.klrsvlambda)

        self.klrsmodel.setObjective(self.klrsvlambda, sense=COPT.MINIMIZE)

    def optimize(self):
        self.klrsmodel.solve()

    def TARGET2LAMB(self):
        self.klrsmodel.solve()
        return self.klrsvlambda.x

    def TARGET2RADIUS(self):
        self.optprob = [0]
        self.eprob =  [0]
        total = 0
        print(self.klrsvlambda.x)
        for n in self.N1:
            self.optprob.append(1/self.samplenum*math.exp(-self.klrsvloss[n].x/self.klrsvlambda.x))
            self.eprob.append(1/self.samplenum)
            total+=self.optprob[n]
        for n in self.N1:
            self.optprob[n] = self.optprob[n]/total

        radius = KL(self.optprob, self.eprob)
        return radius

class TERM:
    def __init__(self, demand, lamb, reward, cost):
        self.samplenum = len(demand)-1
        self.demand = demand
        self.reward = reward
        self.cost = cost
        self.lamb = lamb
        self.N1 = range(1, self.samplenum+1)
        env = cp.Envr()
        self.termmodel = env.createModel('TERM')

        self.termmodel.setParam(COPT.Param.Logging, 0)

        self.termvorder = self.termmodel.addVar(lb=0, vtype=COPT.CONTINUOUS)

        self.termvloss = [0]

        self.termvtarget = self.termmodel.addVar(lb=-COPT.INFINITY, vtype=COPT.CONTINUOUS)

        for n in self.N1:
            self.termvloss.append(self.termmodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vloss[{}]'.format(n)))

        self.termvmin = [0]

        for n in self.N1:
            self.termvmin.append(
                self.termmodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                     name='vmin[{}]'.format(n)))

        self.termmodel.addConstrs(self.termvmin[n] <= self.termvorder for n in self.N1)
        self.termmodel.addConstrs(self.termvmin[n] <= self.demand[n] for n in self.N1)

        self.termcloss = [0]
        for n in self.N1:
            self.termcloss.append(
                self.termmodel.addConstr(-self.reward * self.termvmin[n] + cost * self.termvorder == self.termvloss[n],
                                        name='closs[{}]'.format(n)))

        self.termvu = [0]
        for n in self.N1:
            self.termvu.append(self.termmodel.addVar(lb=-COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vu[{}]'.format(n)))

        for n in self.N1:
            self.termmodel.addConstr(self.termvu[n]==self.termvloss[n]-self.termvtarget)

        self.termvexp = [0]
        for n in self.N1:
            self.termvexp.append(self.termmodel.addVar(lb=0, ub=COPT.INFINITY, name='vexp[{}]'.format(n)))

        self.termvlamb= self.termmodel.addVar(lb=self.lamb, ub=self.lamb, name = 'vlamb[]')

        for n in self.N1:
            self.termmodel.addExpCone([self.termvexp[n], self.termvlamb, self.termvu[n]], COPT.EXPCONE_PRIMAL)

        self.termmodel.addConstr(cp.quicksum(self.termvexp[n]/self.samplenum for n in self.N1)==self.lamb)

        self.termmodel.setObjective(self.termvtarget, sense=COPT.MINIMIZE)

    def optimize(self):
        self.termmodel.solve()

    def LAMB2TARGET(self):
        self.termmodel.solve()
        return self.termmodel.objVal

    def LAMB2RADIUS(self):
        self.termmodel.solve()
        self.optprob = [0]
        self.eprob = [0]
        total = 0
        for n in self.N1:
            self.optprob.append(1 / self.samplenum * math.exp(-self.termvloss[n].x / self.lamb))
            self.eprob.append(1 / self.samplenum)
            total += self.optprob[n]
        for n in self.N1:
            self.optprob[n] = self.optprob[n] / total

        radius = KL(self.optprob, self.eprob)
        return radius

class KLDRO:
    def __init__(self, demand, radius, reward, cost):
        self.demand = demand
        self.radius = radius
        self.reward = reward
        self.cost = cost
        self.samplenum = len(demand)-1
        self.N1 = range(1, self.samplenum+1)
        env = cp.Envr()
        self.kldromodel = env.createModel('KLDRO')

        self.kldromodel.setParam(COPT.Param.Logging, 0)

        self.kldrovorder = self.kldromodel.addVar(lb=0, vtype=COPT.CONTINUOUS, name='vorder')

        self.kldrovlamb = self.kldromodel.addVar(lb=0, vtype=COPT.CONTINUOUS, name='vlamb')

        self.kldrovtau = self.kldromodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype= COPT.CONTINUOUS, name='vtau')

        self.kldrovloss = [0]
        for n in self.N1:
            self.kldrovloss.append(self.kldromodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                                        name='vloss[{}]'.format(n)))

        self.kldrovmin = [0]

        for n in self.N1:
            self.kldrovmin.append(
                self.kldromodel.addVar(lb=-COPT.INFINITY, ub=+COPT.INFINITY, vtype=COPT.CONTINUOUS,
                                      name='vmin[{}]'.format(n)))

        self.kldromodel.addConstrs(self.kldrovmin[n] <= self.kldrovorder for n in self.N1)
        self.kldromodel.addConstrs(self.kldrovmin[n] <= self.demand[n] for n in self.N1)

        self.kldrocloss = [0]
        for n in self.N1:
            self.kldrocloss.append(
                self.kldromodel.addConstr(-self.reward * self.kldrovmin[n] + cost * self.kldrovorder == self.kldrovloss[n],
                                         name='closs[{}]'.format(n)))

        self.kldrovu = [0]
        for n in self.N1:
            self.kldrovu.append(self.kldromodel.addVar(lb=-COPT.INFINITY, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS, name='vu[{}]'.format(n)))

        for n in self.N1:
            self.kldromodel.addConstr(self.kldrovu[n]==self.kldrovloss[n]-self.kldrovtau)

        self.kldrovexp = [0]

        for n in self.N1:
            self.kldrovexp.append(self.kldromodel.addVar(lb=0, vtype=COPT.CONTINUOUS, name='vexp[{}]'.format(n)))

        for n in self.N1:
            self.kldromodel.addExpCone([self.kldrovexp[n], self.kldrovlamb, self.kldrovu[n]], COPT.EXPCONE_PRIMAL)

        self.kldromodel.addConstr(cp.quicksum(self.kldrovexp[n]/self.samplenum for n in self.N1)==self.kldrovlamb)

        self.kldromodel.setObjective(self.kldrovtau+self.kldrovlamb*self.radius, COPT.MINIMIZE)

    def optimize(self):
        self.kldromodel.solve()


    def RADIUS2TARGET(self):
        self.kldromodel.solve()
        return -self.kldromodel.objVal+self.radius*self.kldrovlamb

    def RADIUS2LAMB(self):
        self.kldromodel.solve()
        return self.kldrovlamb.x






