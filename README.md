# Physics-Informed-Neural-Network:No-Autograd
Learning attempt, fitting sinx over a larger set of x using a PINN.

# Requirements
Numpy,
Matplotlib.

# Features
4 Layer MLP,
Manual Backpropagation,
Purely Numpy Based,
PINN integrated.

# Model
The Network is initialised with manual weights and biases and essentially every layer uses the tanh function as the activation, so:
a = tanh(wx+b)

The physics informed part uses the differential equation, dy/dx + y = sinx + cosx that has a solution of y = sinx for y(0) = 0, so we are fitting for the function sinx.

The physics residual is dy/dx + y - (sinx + cosx). 

The total loss is L = 0.5*((output-y)**2 + (dy/dx + y - (sinx + cosx))**2).

The backpropagation is essentially a sort of SGD as every data point of the dataset, one by one is used to train the Neural Network.

Then the training loop is as follows,
Forward pass,
Loss,
Back Propagation,
Update.

This model is an educational project, and is not meant to be efficient.
