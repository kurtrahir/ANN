# Solving MNIST Handwritten Digit Recognition using convolutional neural networks built from scratch using cupy.


## Problem Definition

This project was an exercise combining Software Engineering and Machine Learning to build a custom package allowing for the construction of artificial neural networks (ANN). The idea behind it was to implement those things necessary for the construction of a decently performing model on the MNIST handwritten digit dataset. The implementation should take the form of a package and be extensible in the future.

### Requirements

To build a CNN the following items were designated as _required_.

- Convolutional Layers
- ReLu Activation function
- Densely Connected Layer
- Softmax Activation
- Categorical Cross Entropy Loss
- Stochastic Gradient Descent

To build a model that performed well the following _optional_ items were added.

- Batch Normalization
- Data Augmentation
- Max Pooling Layers
- Adam Optimizer

## Layers

Neural Networks are typically made up of one or more layers. They usually apply some sort of non-linearity to a linear transformation of the input that is fed to it. This linear transformation is typically learned during a *training* phase, where the layers **weights** and sometimes **bias** are optimized to reduce a certain loss metric.

### Densely connected layers.
Densely connected layers are the most basic and widely used layers in the field of ANN.

They typically consist of a certain number of neurons, who all receive the same input and learn a linear transformation which is then passed through an activation function, allowing the network to learn non-linear relationships. Given input $\mathbf{I} = [I_0, I_1, ..., I_n]$ a neuron with weights $\mathbf{w}= \left [ \begin{matrix}w_0 \\ w_1 \\ ... \\ w_n \end{matrix} \right]$, bias $b$ and activation function $\sigma(x)$ will compute $\sigma(\mathbf{w} \cdot I + b)$.

A densely connected layer with S neurons can be expressed using a weight matrix W consisting of S weight vectors $\mathbf{W} = [\mathbf{w_0},\mathbf{w_1},...,\mathbf{w_S}] = \left [\begin{matrix} w_{00} & w_{10} & ... & w_{S0} \\ w_{01} & w_{11} & ... & w_{S1} \\ ...& ...& ...& ... \\ w_{0n} & w_{1n} & ... & w_{Sn} \end{matrix}\right ]$ and a bias vector $\mathbf{b} = [b_0,b_1,...,b_S]$
