# Physics-Informed-Neural-Network vs MLP: No Autograd
Comparing a PINN on small dataset, a NN on large dataset and small dataset.

# Requirements
Numpy,
Matplotlib.

# Features
MLP: 1-tanh-3-tanh-3-tanh-3-tanh-1-tanh-out-mse,
PINN: 1-tanh-3-tanh-3-tanh-3-tanh-1-tanh-out-mse+physicsloss,
Manual Backpropagation,
Numpy Based.

# Models
## Model 1:
Model 1 is an MLP with tanh as the activation function on all layers execpt last year. It's the number of neurons on every layer of the network is as follows, 1-3-3-3-1-1. Followed by Mean Squared Error used to evaluate the loss.

## Model 2:
Model 2 follows the same skeleton as Model 1. The loss is described by the following equation, L = (1 - lambda) * MSELOSS + lambda * PHYSICSLOSS, where MSELOSS is the mean squared error between y and yhat, lambda is the coefficient that controls the relative importance of MSELOSS and PHYSICSLOSS and PHYSICSLOSS is the physics loss described by the following residual, d^2y/dy^2 + y = 0.

###Note: This experiment is an exploration for learning, and is not built to be efficient.
