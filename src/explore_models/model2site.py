import numpy as np
class Model2site:
    def __init__(self, C, model):
        t_pars = {"IRF":[0,1,0,1],
                  "NFkB":[0,0,1,1],
                  "AND":[0,0,0,1],
                  "OR":[0,1,1,1],}
        self.t = t_pars[model]
        self.beta = np.array([1 for i in range(4)])
        self.beta[3] = C
    def calculateState(self, N, I):
        self.state = np.array([1, I, N, I*N])
        # print(self.state)
    def calculateF(self):
        self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)
        # print(self.f)

def explore_model2site(C, model, N, I):
    m = Model2site(C, model)
    m.calculateState(N, I)
    m.calculateF()
    return m.f

