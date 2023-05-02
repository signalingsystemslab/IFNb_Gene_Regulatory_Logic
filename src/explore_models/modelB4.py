# Author: Allison Schiffman

# Copy p50_modelB3.py but make the class modelB4 with pars.K = parsT[0] and pars.C = parsT[1]

import numpy as np

class ModelB4:
    def __init__(self, parsT):
        self.K = parsT[0]
        self.C = parsT[1]
        self.parsT = parsT[2:len(parsT)]
        self.beta = np.array([1 for i in range(8)])
        # beta position 0 refers to none, beta position 1 refers to Ki1, beta position 2 refers to Ki2, beta position 3 refers to kN,
        # beta position 4 refers to Ki1Ki2, beta position 5 refers to Ki1kN, beta position 6 refers to Ki2kN, beta position 7 refers to Ki1Ki2kN
        # 
        # multiply beta by K at positions referring to Ki2, Ki1Ki2, Ki2kN, and Ki1Ki2kN
        # multiply beta by C at positions referring to Ki1Ki2, Ki1Ki2kN
        self.beta[2] = self.K
        self.beta[4] = self.K * self.C
        self.beta[6] = self.K
        self.beta[7] = self.K * self.C

        self.t = np.hstack((0, self.parsT, 1))

    def calculateState(self, N, I):
        self.state = np.array([1, I, I, N, I**2, I*N, I*N, I**2*N])

    def calculateF(self):
        self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

def explore_modelB4(parsT, N, I):
    model = ModelB4(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f