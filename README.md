# hessian_test
This repository uses [PyHessian](git@github.com:amirgholami/PyHessian.git) to plot the Eigen-Spectrum Density (ESD) and other statistics of the Loss Hessian and Predictor Hessian.

## Modifications to PyHessian:
### Predictor Hessian
To plot the ESD of predictor Hessian, we backpropagate through the output when collecting gradients instead of backpropagating through the loss:
```
outputs[0][self.output_index].backward(create_graph=True)
```
To enable this, set `wrt_loss=False` in the constructor and set the `output_index` to the class of the prediction w.r.t which you want to calculate the gradient with:
```
obj = hessian(model, loss_fn,  data=(traj_ip, traj_labels),wrt_loss=False,output_index = 0)
```
Here `traj_ip`,`traj_labels` is the input and correct label for a training sample.

### Injecting custom eigen value
To insert a specific eigen-value for sanity check, set the eigen value by passing it as `lamda` to the constructor
```
obj = hessian(model, loss_fn,  data=(traj_ip, traj_labels),wrt_loss=False,output_index = 0,lamda=200)
```
## Datasets:
1. Gauss-2 : 100 samples each from 2D gaussians with mean [0,100] and variance [1].
2. Cifar2c : 2 classes from Cifar-10.

## Usage
To run the example with ResNet-18 and 2 classes of Cifar10:
```
python main_resnet_cifar.py
```
To test the code by adding additional EigenValue 
```
python main_resnet_cifar_custom_ev.py
```

