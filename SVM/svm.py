'''
 # @ Author: Chen Liqian
 # @ Create Time: 2023-11-02 15:48:34
 # @ Modified by: Chen Liqian
 # @ Modified time: 2023-11-02 15:48:36
 # @ Description:define the SVM class
 '''
import kernel as k
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cvxopt.solvers

class SVM(object):
    def __init__(self, kernel):
        self.kernel = k.Kernel(kernel)

    def _check_kernel(self):
        if self.kernel.get_kernel_type() not in k.KERNEL_TYPE:
            print("unavailable kernel!\n")
            return False
        else:
            return True
            
    def _generate_samples(self, sample_cnt, feature_cnt):
        samples = np.matrix(np.random.normal(size=feature_cnt * sample_cnt)).reshape(sample_cnt, feature_cnt)
        return samples
    
    def _label_samples(self, sample, feature_cnt):
        if(feature_cnt == 2):
            labels = 2 * (sample.sum(axis=1) > 0) - 1.0
        # TODO: multi-class labels
        else:
            pass
        return labels

    def _display_samples(self, sample, label):
        x = sample[:, 0].tolist()
        y = sample[:, 1].tolist()
        plt.scatter(x, y,c=label.tolist(), cmap=cm.Paired)
        plt.savefig("samples.pdf")

    def _deploy_kernel_method(self, sample):
        sample_cnt, feature_cnt = sample.shape
        kernel_matrix = np.zeros((sample_cnt, sample_cnt))
        for i, x_i in enumerate(sample):
            for j, x_j in enumerate(sample):
                kernel_matrix[i, j] = self.kernel.kernel(x_i, x_j)
        
        # for i, x_i in enumerate(sample):
        #     for j, x_j in enumerate(sample):
        #         print(kernel_matrix[i, j])

            # print("\n")    

        return kernel_matrix
    
    def train(self, sample, label):
        sample_cnt, feature_cnt = sample.shape
        K = self._deploy_kernel_method(sample)
        

    def fit(self, sample_cnt, feature_cnt):

        if self._check_kernel():
            samples = self._generate_samples(sample_cnt=sample_cnt, feature_cnt=feature_cnt)
            labels = self._label_samples(sample=samples, feature_cnt=feature_cnt)
            self._display_samples(samples, labels)
           
            self.train(sample=samples, label=labels)

