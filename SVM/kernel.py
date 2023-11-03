'''
 # @ Author: Chen Liqian
 # @ Create Time: 2023-11-02 14:59:11
 # @ Modified by: Chen Liqian
 # @ Modified time: 2023-11-02 15:02:40
 # @ Description:implement some common and popular kernel methods used in SVM:
                - linear kernel
                - polynomial kernel
                - Gaussian kernel
                - RBF(Gaussian Radial Basis Function)
                - Laplace RBF kernel
                - Bessel Function kernel
                - Hyperbolic Tangent kernel
                - Sigmoid kernel
                - ANOVA kernel
 '''

import numpy as np

KERNEL_TYPE = ['linear', 'polynomial', 'gassuian', 'RBF', 'laplace-RBF', 'bessel-function', 'hyperbolic tangent', 'sigmoid', 'ANOVA']

class Kernel(object):
    def __init__(self, kernel_type) -> None:
        self.kernel_type = kernel_type
    
    def kernel(self, x, y):
        if self.kernel_type == 'linear':
            return np.inner(x, y)
        if self.kernel_type == 'polynomial':
            pass
        if self.kernel_type == 'gassuian':
            pass
        if self.kernel_type == 'RBF':
            pass
        if self.kernel_type == 'laplace-RBF':
            pass
        if self.kernel_type == 'bessel-function':
            pass
        if self.kernel_type == 'hyperbolic tangent':
            pass
        if self.kernel_type == 'sigmoid':
            pass
        if self.kernel_type == 'ANOVA':
            pass

    def get_kernel_type(self):
        return self.kernel_type