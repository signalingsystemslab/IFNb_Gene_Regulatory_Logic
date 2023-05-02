# Author: Allison Schiffman

import numpy as np

class ModelB2:
    def __init__(self, parsT):
        self.K = parsT[0]
        self.parsT = parsT[1:len(parsT)]
        self.beta = np.array([1 for i in range(8)])
        # beta position 0 refers to none, beta position 1 refers to Ki1, beta position 2 refers to Ki2, beta position 3 refers to kN,
        # beta position 4 refers to Ki1Ki2, beta position 5 refers to Ki1kN, beta position 6 refers to Ki2kN, beta position 7 refers to Ki1Ki2kN
        # 
        # multiply beta by K at positions referring to Ki2, Ki1Ki2, Ki2kN, and Ki1Ki2kN
        # multiply beta by C at positions referring to Ki1Ki2, Ki1Ki2kN
        self.beta[2] = self.K
        self.beta[4] = self.K
        self.beta[6] = self.K
        self.beta[7] = self.K

        # print(self.parsT)
        self.t = np.hstack((0, self.parsT, 1))
        # print("t= " + str(self.t))

    def calculateState(self, N, I):
        self.state = np.array([1, I, I, N, I**2, I*N, I*N, I**2*N])
        # print("state = "+ str(self.state))
        # print(self.beta*self.t)

    def calculateF(self):
        self.f = np.dot(np.transpose(self.state),(self.beta * self.t)) / np.dot(np.transpose(self.state), self.beta)

def explore_modelB2(parsT, N, I):
    model = ModelB2(parsT)
    model.calculateState(N, I)
    model.calculateF()
    return model.f
# def main():
#     # initialize modelB2 with parsT = 1.51644916   0.16063854   0.12538234   0.03271724   0.83962291   0.46298051   0.49638786 
#     p50 = ModelB2([1.51644916, 0.16063854, 0.12538234, 0.03271724, 0.83962291, 0.46298051, 0.49638786])
#     p50.calculateState(0, 0.25)
#     p50.calculateF()
#     print(p50.f)
#     # Load the dictionary 'p50_dict.npy' and add the following key-value pairs: 'B2': p50.f 
#     p50_dict = np.load('p50_dict.npy',allow_pickle=True).item()
#     p50_dict['B2'] = p50.f
#     # print p50_dict in a pretty way
#     for key, value in p50_dict.items():
#         print(key, value)
#     # Save the dictionary 'p50_dict' to disk
#     np.save('p50_dict.npy', p50_dict)


# if __name__ == '__main__':
#     main()