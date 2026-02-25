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
The Network is initialised with manual weights and biases. Every layer uses the tanh function as the activation: a = tanh(wx+b)

The Physics Informed Residual is dy/dx + y - (sinx + cosx). The Residual's differential equation form's solution is y = sinx for y(0) = 0, fitting for the function sinx.

The total loss is L = 0.5*((output-y)**2 + (dy/dx + y - (sinx + cosx))**2).

The backpropagation is a sort of SGD where every data point of the dataset, one by one, is used to train the Neural Network.

The training loop is as follows:
Forward pass,
Loss Calculation,
Back Propagation,
Update.

This model is an educational exploration, and is not built to be efficient.
