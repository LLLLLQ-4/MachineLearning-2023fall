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
from sklearn.svm import LinearSVC

class SVM(object):
    def __init__(self, kernel, sample_cnt, feature_cnt):
        self.kernel = k.Kernel(kernel)
        self.sample_cnt = sample_cnt
        self.feature_cnt = feature_cnt

    def _check_kernel(self):
        if self.kernel.get_kernel_type() not in k.KERNEL_TYPE:
            print("unavailable kernel!\n")
            return False
        else:
            return True
            
    def _generate_samples(self):
        samples = np.matrix(np.random.normal(size=self.feature_cnt * self.sample_cnt)).reshape(self.sample_cnt, self.feature_cnt)
        return samples
    
    def _label_samples(self, sample):
        if(self.feature_cnt == 2):
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

    def _compute_kernel_matrix(self, sample):
        kernel_matrix = np.zeros((self.sample_cnt, self.sample_cnt))
        for i, x_i in enumerate(sample):
            for j, x_j in enumerate(sample):
                kernel_matrix[i, j] = self.kernel.kernel(x_i, x_j)

        print("kernel-matrix: \n", kernel_matrix)
        return kernel_matrix
    
    def _solve_quadratic_programming(self, label, kernel_matrix):

        # normalization: min (1/2)x^T P x + q^T x (min (1/2)α^T H α - 1^T α)
        P = cvxopt.matrix(np.outer(label, label) * kernel_matrix)
        print("P: \n", P)
        
        q = cvxopt.matrix(-1.0 * np.ones(self.sample_cnt))

        # s.t. G x <= h (-a_i <= 0)
        G = cvxopt.matrix(np.diag(np.ones(self.sample_cnt) * -1.0))
        h = cvxopt.matrix(np.zeros(self.sample_cnt))

        # s.t. A x = b (y^T α = 0)
        A = cvxopt.matrix(label, (1, self.sample_cnt))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b, kktsolver="chol")

        return np.ravel(solution['x'])


    def _compute_lagrange_multipliers(self, sample, label):
        kernel_matrix = self._compute_kernel_matrix(sample)
        
        return self._solve_quadratic_programming(label, kernel_matrix)
    
    def _result_SVM_model(self, sample, label, lagrange_mutlipiler):
        w = np.dot((label * lagrange_mutlipiler).T, sample)[0]

        S = (lagrange_mutlipiler > 1e-5).flatten()
        b = np.mean(label[S] - np.dot(sample[S], w.reshape(-1,1)))
        
        print("W: ", w)
        print("\n")
        print("b: ", b)

    def _display_train(self):
        pass

    def train(self, sample, label):
        lagrange_multipliers = self._compute_lagrange_multipliers(sample, label)
        self._result_SVM_model(sample, label, lagrange_multipliers)

    def fit(self):

        if self._check_kernel():
            samples = self._generate_samples()
            labels = self._label_samples(sample=samples)
           
            self._display_samples(samples, labels)
            model_linear = LinearSVC(C=0.1, loss='squared_hinge', dual=True)
            model_linear.fit(samples, labels)
            print("W: \n", model_linear.coef_[0])
            print("\nb: ", model_linear.intercept_[0])

            x_min, x_max = samples[:, 0].min() - 0.2, samples[:, 0].max() + 0.2
            y_min, y_max = samples[:, 1].min() - 0.2, samples[:, 1].max() + 0.2
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                                np.arange(y_min, y_max, 0.002))

            test_point = np.c_[xx.ravel(), yy.ravel()]
            predict = model_linear.predict(test_point)
            zz = predict.reshape(xx.shape)
            plt.contourf(xx, yy, zz, cmap = cm.Paired, alpha=0.6)
            x = samples[:, 0].tolist()
            y = samples[:, 1].tolist()
            plt.scatter(x, y,c=labels.tolist(), cmap=cm.Paired)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig("result.pdf")

            # plt.scatter(samples[y==0,0], samples[])
            # self.train(sample=samples, label=labels)

            # self._display_train()

