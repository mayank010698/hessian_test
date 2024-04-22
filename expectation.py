
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

def expectation(density,grids):
    shift = grids[1]-grids[0]
    return 0.5*np.sum((np.abs(grids+shift)+np.abs(grids))*density)*shift


def plot_eigen_value():
    pass


folder_name = "/projects/dali/mayanks4/hessian_test/2024-04-18_23-09-31"
density_grids = {}
# eigen_values_list = []

with open(f'{folder_name}/predictor_density_grids.pkl',"rb") as f:
    density_grids = pkl.load(f)
# 
with open(f'{folder_name}/predictor_eigen_values_dict.pkl','rb') as f:
    predictor_eigen_values_dict = pkl.load(f)

indices = 1
epochs = 50

abs_expectation = []
max_eiegen_val  = []

for epoch in range(0,epochs,5):
    for index in range(indices):
        density = density_grids[epoch]["density"]
        grids   = density_grids[epoch]["grids"]
        abs_expectation.append(expectation(density,grids))
        

plt.figure()
plt.plot(range(0,epochs,5),abs_expectation,label=r"$\mathbb{E}[ \vert \Lambda_i \vert]$")

for epoch in range(0,epochs,5):
    plt.scatter(2*[epoch],predictor_eigen_values_dict[epoch])
   
plt.legend()
plt.xlabel('Epoch')

plt.title(f'Predictor EigenValue')
plt.savefig('{}/predictor_ev.pdf'.format(folder_name))  
plt.close()


