'''
Different non-linear functions to project data from a 2D space to a higher dimensional, non-linear space
'''
import numpy as np

class linear_log():

    def __init__(self, H=6):
        linear = np.zeros((2, H))
        np.fill_diagonal(linear, 1)
        self.linear = linear

    def __call__(self, X):
        '''
            X: np.array [Batch x 2]
            returns: np.array [Batch x H]
        '''
        return np.log(X, self.linear)

class linear_exp():

    def __init__(self, H=6):

        linear = np.zeros((2, H))
        np.fill_diagonal(linear, 1)
        self.linear = linear
        
        linear_inverse = np.zeros((H,2))
        np.fill_diagonal(linear_inverse, 1)
        self.linear_inverse = linear_inverse


    def __call__(self, X):
        '''
        Forward pass
            X: np.array [Batch x 2]
            returns: np.array [Batch x H]
        '''
        return np.exp(np.matmul(X, self.linear))
    
    def inverse(self, X):
        '''
        Inverse
            X: np.array [Batch x H]
            returns: np.array [Batch x 2]
        '''
        return np.matmul(np.log(X), self.linear_inverse)


