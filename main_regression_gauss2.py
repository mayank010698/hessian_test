# %%
from models.dense_layers import Regression
import torchvision.datasets
import torch
import os
from torchvision import transforms
import matplotlib.pyplot as plt

from pyhessian.hessian import hessian
from pyhessian import utils
from density_plot import get_esd_plot
import numpy as np
from datetime import datetime

import pickle as pkl


from dataloaders.gauss2 import Gauss2
import json

############### CONFIG #####################################
lr          = 1e-3
wd          = 1e-5
model_name  = 'Regression'
dataset     = 'Gauss2'
meta = "Sanity check using a single layer regression problem "
epochs      = 50

import torch.nn.functional as F
def loss_fn(x,y):
    return torch.nn.MSELoss()(x,y)

# loss_fn     = torch.nn.CrossEntropyLoss()
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
n_eigen     = 2

current_datetime = datetime.now()
folder_name      = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(folder_name,exist_ok=True)

config_dict = {}
config_dict["lr"] = lr
config_dict["wd"] = wd
config_dict["model_name"] = model_name
config_dict["dataset"] = dataset
config_dict["epochs"] = epochs


print(json.dumps(config_dict))
with open(f"{folder_name}/config.txt","w") as f:
    f.write(json.dumps(config_dict))

################ LOAD DATA #################################


with open("gauss2.pkl","rb") as f:
    data_dict = pkl.load(f)
    train_data_list = data_dict



train_data = Gauss2(train_data_list)
test_data = Gauss2(train_data_list)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=100,shuffle=True)

num_classes = 1

loss_traj_data = iter(test_data_loader)
loss_traj_batch = next(loss_traj_data)

################# METRICS ###################################

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

# predictor_trace_list = []  
loss_trace_list = [] 
loss_eigen_values_dict = {} 
# predictor_eigen_values_dict = {}



batch_loss_list = []
# predictor_density_grids = {}
loss_density_grids = {}
################ MODEL & OPTIMIZER #######################################

model = Regression() 
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)

################ TRAIN #######################################

for epoch in range(epochs):
    train_acc = 0
    train_loss = 0
    
    for i,(ip,labels) in enumerate(train_data_loader):
        ip = ip.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(dim=1)
        labels = labels.to(torch.float32)

        optimizer.zero_grad()
        # import pdb
        # pdb.set_trace()
        outputs = model(ip)
        loss = loss_fn(outputs,labels)
        
        batch_loss = loss.item()
        
        loss.backward()
        optimizer.step()

        train_loss += batch_loss
        # preds = torch.argmax(outputs,dim=1)
        preds = outputs > 0.5
        train_acc += torch.sum(preds==labels)

        batch_loss_list.append(batch_loss)

        if(i%10==0):
            print("\tEpoch:{},Batch:{},Loss:{}".format(epoch,i,batch_loss))
            
        
 
    loss_density_grids[epoch] = {}
    
    
    if((epoch)%5==0):
        # idx = np.randint(0,len(train_data))
        loss_hessian_comp = hessian(model, loss_fn,  data=loss_traj_batch, wrt_loss=True,output_index = None)

        loss_eigenvalues, _                    = loss_hessian_comp.eigenvalues(top_n = n_eigen)

        loss_eigen, loss_weight = loss_hessian_comp.density()
        hessian_spectrum_plot             = f"{folder_name}/Epoch:{epoch} wrt_Loss"
        loss_density, loss_grids = get_esd_plot(loss_eigen, loss_weight, hessian_spectrum_plot)
        loss_density_grids[epoch]["density"] = loss_density
        loss_density_grids[epoch]["grids"] = loss_grids
        loss_trc                               = np.mean(loss_hessian_comp.trace())

        loss_trace_list.append(loss_trc)
        loss_eigen_values_dict[epoch] =  loss_eigenvalues

    model.train()
    model.zero_grad()
            
    with torch.no_grad():
        test_acc = 0
        test_loss = 0
        for i,(ip,labels) in enumerate(test_data_loader):
            ip = ip.to(device)
            labels = labels.to(device)

            labels = labels.unsqueeze(dim=1)
            labels = labels.to(torch.float32)

            outputs = model(ip)
            loss = loss_fn(outputs,labels)

            preds = outputs > 0.5

            test_acc += torch.sum(preds==labels)
            test_loss += loss.item()

                
                
        print("Epoch:{},Train Loss:{}, Train Accuracy:{}".format(epoch,train_loss/len(train_data_loader),train_acc/len(train_data)))
        print("Epoch{},Test Loss:{},Test Accuracy:{}".format(epoch,test_loss/len(test_data_loader),test_acc/len(test_data)))

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss/len(train_data_loader))
        test_loss_list.append(test_loss/len(test_data_loader))






with open(f'{folder_name}/loss_density_grids.pkl',"wb") as f:
    pkl.dump(loss_density_grids,f)

with open(f'{folder_name}/loss_eigen_values_dict.pkl',"wb") as f:
    pkl.dump(loss_eigen_values_dict,f)





plt.figure()
plt.plot(batch_loss_list,label='lr:{},wd:{}'.format(lr,wd))
plt.title('Model:{}_Dataset:{}'.format(model_name,dataset))
plt.ylabel('Loss')
plt.savefig('{}/Loss.pdf'.format(folder_name,i))
plt.close()


plt.figure()
plt.plot(loss_trace_list)
plt.xlabel('Epoch')
plt.ylabel('Trace')
plt.title(f'Loss Hessian Trace')
plt.savefig('{}/loss_hessian_trace.pdf'.format(folder_name))
plt.show()
plt.close()

plt.figure()
for epoch in range(0,epochs,5):
    plt.scatter(2*[epoch],loss_eigen_values_dict[epoch])


plt.xlabel('Epoch')
plt.ylabel('EV')
plt.title(f'Loss EigenValue')
plt.savefig('{}/loss_ev.pdf'.format(folder_name))  
plt.close()


def expectation(density,grids):
    shift = grids[1]-grids[0]
    return 0.5*np.sum((np.abs(grids+shift)+np.abs(grids))*density)*shift

abs_expectation = []
for epoch in range(0,epochs,5):
    density = loss_density_grids[epoch]["density"]
    grids   = loss_density_grids[epoch]["grids"]
    abs_expectation.append(expectation(density,grids))
        

plt.figure()
plt.plot(range(0,epochs,5),abs_expectation,label=r"$\sum_{i=1}^p \vert \Lambda_i \vert$")
plt.xlabel('Epoch')
plt.savefig(f"{folder_name}/loss_abs_ev_sum.pdf")
plt.close()


  




