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
