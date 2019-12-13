import numpy as np
import math

class FourierBasis():
    """Fourier Basis to do Basis Expansion for continuous state"""
    def __init__(self, inputDimension, iOrder, dOrder):
        self.inputDimension = inputDimension				# Copy over the provided arguments
        # Compute the total number of terms
        iTerms = iOrder*inputDimension						# Number of independent terms
        dTerms = np.power(dOrder + 1, inputDimension)			# Number of dependent terms
        oTerms = min(iOrder, dOrder)*inputDimension		    # Overlap of iTerms and dTerms
        self.nTerms = iTerms + dTerms - oTerms
        # Initialize c
        self.c = np.zeros((self.nTerms, inputDimension))
        counter = np.zeros(inputDimension)
        for termCount in range(dTerms):                     # First add the dependent terms
            self.c[termCount, :] = counter
            self._increment_counter(counter, dOrder)
        
        termCount = dTerms
        # for (i = 0 i < inputDimension i++) :				
        for i in range(inputDimension):                    # Add the independent terms
            for j in range(dOrder+1, iOrder+1):
                self.c[termCount, :] = np.zeros(inputDimension)
                self.c[termCount,i] = j
                termCount += 1
        #     for (j = dOrder + 1 j <= iOrder j++) :
        #         c[termCount] = vector<double>(inputDimension, 0.0)
        #         c[termCount][i] = (double)j
        #         termCount++

    def get_num_outputs(self):
        return self.nTerms

    def basify(self, x):
        ret = np.zeros(self.nTerms)
        for i in range(self.nTerms):
            ret[i] = math.cos(math.pi * np.dot(self.c[i], x))
        return ret

    def _increment_counter(self, buff, maxDigit):
        for i in range(len(buff)):
            buff[i] += 1
            if buff[i] <= maxDigit:
                break
            buff[i] = 0


# stateDim = 2
# iOrder = 1
# dOrder = 1

# fb = FourierBasis(stateDim, iOrder, dOrder)

# testArr = np.zeros(stateDim)
# testArr[0] = 0.41176470588235292
# testArr[1] = 0.5

# res = fb.basify(testArr)
# print(res)