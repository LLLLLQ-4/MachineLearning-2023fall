'''
 # @ Author: Chen Liqian
 # @ Create Time: 2023-11-02 15:32:09
 # @ Modified by: Chen Liqian
 # @ Modified time: 2023-11-02 15:34:30
 # @ Description:test our implement of svm

 '''
import kernel
from svm import SVM
import argh

import matplotlib.pyplot as plt

def display_result():
    pass

def handler(*, num_samples=10, num_features=2, grid_size=20, filename="svm.pdf"):
    svm = SVM()
    svm.fit(num_samples, num_features, grid_size, filename)

    display_result()

    
if __name__ == "__main__":
    argh.dispatch_command(handler)
    


