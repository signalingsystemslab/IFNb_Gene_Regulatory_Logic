# Author: Allison Schiffman
#
# import numpy  
import numpy as np

class ModelB3:
    def __init__(self, parsT):
        self.C = parsT[0]
        self.parsT = parsT[1:len(parsT)]
        self.beta = [1 for i in range(8)]
        # beta position 0 refers to none, beta position 1 refers to Ki1, beta position 2 refers to Ki2, beta position 3 refers to kN,
        # beta position 4 refers to Ki1Ki2, beta position 5 refers to Ki1kN, beta position 6 refers to Ki2kN, beta position 7 refers to Ki1Ki2kN
        # multiply beta by C at positions referring to Ki1Ki2, Ki1Ki2kN
        self.beta[4] = self.C
        self.beta[7] = self.C
        
        self.t = [0] + self.parsT + [1]

    def calculateState(self, N, I):
        self.state = [1, I, I, N, I**2, I*N, I*N, I**2*N]

    def calculateF(self):
        self.f = np.transpose(self.state) * np.dot(self.beta, self.t) / np.dot(np.transpose(self.state), self.beta)

def main():
    # initialize modelB3 with parsT = -0.266231648   0.125798719   0.123052470   0.007045496   0.447518199   0.523644773   0.530709398
    p50 = ModelB3([-0.266231648, 0.125798719, 0.123052470, 0.007045496, 0.447518199, 0.523644773, 0.530709398])
    p50.calculateState(0, 0.25)
    p50.calculateF()
    print(p50.f)
    # Load the dictionary 'p50_dict.npy' and add the following key-value pairs: 'B3': p50.f
    p50_dict = np.load('p50_dict.npy',allow_pickle=True).item()
    p50_dict['B3'] = p50.f
    # print p50_dict in a pretty way
    for key, value in p50_dict.items():
        print(key, value)
    # Save the dictionary 'p50_dict' to disk
    np.save('p50_dict.npy', p50_dict)

if __name__ == '__main__':
    main()