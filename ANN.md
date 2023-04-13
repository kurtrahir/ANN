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

### __Densely connected layers__
Densely connected layers are the most basic and widely used layers in the field of ANN.
#### __Structure and function__
They typically consist of a certain number of neurons, who all receive the same input and learn a linear transformation which is then passed through an activation function, allowing the network to learn non-linear relationships. Given input $\mathbf{I} = [I_0, I_1, ..., I_n]$ a neuron with weights $\mathbf{w}= \left [ \begin{matrix}w_0 \\ w_1 \\ ... \\ w_n \end{matrix} \right]$, bias $b$ and activation function $\sigma(x)$ will compute $\sigma(\mathbf{w} \cdot I + b)$.

A densely connected layer with S neurons can be expressed using a weight matrix W consisting of S weight vectors $\mathbf{W} = [\mathbf{w_0},\mathbf{w_1},...,\mathbf{w_n}] = \left [\begin{matrix} w_{00} & w_{10} & ... & w_{S0} \\ w_{01} & w_{11} & ... & w_{S1} \\ ...& ...& ...& ... \\ w_{0n} & w_{1n} & ... & w_{Sn} \end{matrix}\right ]$ and a bias vector $\mathbf{b} = [b_0,b_1,...,b_S]$

This allows the activation of the dense layer to be expressed as: $a = \sigma(\mathbf{W\cdot I + b} )$.

A barebones example of a dense layer is therefore:

```
class Dense(Layer):
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.inputs = None

    def forward(inputs):
        self.inputs = inputs #store inputs for backwards pass
        return self.activation.forward(
            cp.dot(inputs, self.weights) + self.bias
        )
```
#### __Gradients__

Neural networks are only useful because of their ability to learn. This relies on the calculation of gradients of a certain performance metric, typically a loss function. This loss quantifies how "wrong" the current output of the network is. By finding the gradient of this loss function with regards to the trainable parameters of the network, we can adjust the parameters to decrease the value of this loss.

Consider input and target output pair $\mathbf{I}$, $\mathbf{O}$. Prediction $\mathbf{P}$ of the network being $\mathbf{P} = a = \sigma(\mathbf{\mathcal{z}} )$ where $\mathcal{z}=\mathbf{W\cdot I + b}$.
The loss function $\mathcal{L}$ takes in the target output and the prediction to quantify the quality of the prediction: $\mathcal{L}(\mathbf{P},\mathbf{O})$.

Finding the gradients for the trainable parameters is achieved by using the chain rule. ${d\mathcal{L}\over d\mathbf{W}} = {d\mathcal{L}\over da} \cdot {da \over d\mathcal{z}} \cdot {d\mathcal{z} \over d\mathbf{W}} $ and similarly ${d\mathcal{L}\over d\mathbf{b}} = {d\mathcal{L}\over da} \cdot {da \over d\mathcal{z}} \cdot {d\mathcal{z} \over d\mathbf{b}}$.

This makes sense for the last layer in the neural network. If the layer is earlier in the the net, we can _backpropagate_ the loss. Consider the case of a two layer network. The output layer is at index ($L$) and the input layer is at index ($L-1$)

Let $a_{L-1}$ be the activation value of layer $L-1$. We have $a_{L-1} = \sigma(\mathcal{z}_{L-1})=\sigma(\mathbf{W}_{L-1}\cdot I + \mathbf{b}_{L-1})$ and $a_L = \sigma(\mathcal{z}_L)=\sigma(\mathbf{W}_L\cdot a_{L-1} + \mathbf{b}_L)$.

Continuing with the chain rule we can obtain the gradients with regards to the input layer's parameters.
${d\mathcal{L} \over d\mathbf{W}_{L-1}} = {d\mathcal{L}\over da_L} \cdot {da_L \over d\mathcal{z}_L} \cdot {d\mathcal{z}_L \over da_{L-1}} \cdot {{da_{L-1}} \over d\mathcal{z}_{L-1}} \cdot {d\mathcal{z}_{L-1} \over d\mathbf{W}_{L-1}}$ with the same being applicable to $\mathbf{b}_{L-1}$ and $\mathbf{I}$.

This is extensible to a network with any number of layers L, simply backpropagate the gradient with regards to the input of a layer to the previous layer until the entire network has been traversed.

We know ${d\mathcal{z} \over d\mathbf{W}} = I$ and ${d\mathcal{z} \over d\mathbf{b}} = 1$ for all layers.

If we provide the gradient to be propagated to the backwards method, we can write it in a general purpose format (ignoring whether the layer in question is an output, hidden or input layer).

The barebones implementation for the dense layer gradient caculation  is therefore:

```
def backward(gradient):
    d_activation = self.activation.backward(
        gradient
    )
    self.d_bias = d_activation
    self.d_weights = cp.dot(
        self.inputs.T,
        d_activation
    )
    d_inputs = cp.dot(
        d_activation,
        self.weights.T
    )
    return d_inputs
```

### __Convolutional Layer__

In the dense layers we just reviewed, each neuron learns a weight for every input value plus a bias. This means for $n$ input values and $S$ neurons, a layer will learn $S(n+1)$ parameters. For many applications this may be fine, however when the input consists of images, the number parameters to be learned can explode. For example: for a 256x256 image, where each pixel consists of 3 color channels (RGB), using an input layer with 128 neurons we get $(256\times 256 \times 3 + 1) \times 128 = 25,165,952$ parameters.

Convolutional layers aim to reduce the amount of parameters learned by sharing them. This is done through the use of kernels, whose sizes may vary, but typically are used in the shapes of $3\times 3$, $5\times 5$ or $7\times 7$.

These kernels are passed over the image computing the dot product between the kernel and the subregion (or receptive field) being considered.

With input $\mathbf{I}$ of size $(x,y)$ and kernel $\mathbf{K}$ of size $(k_x,k_y)$, the output $\mathbf{O}$ of the convolutional layer at index $(i,j)$ is:

 $\mathbf{O}(i,j) = \sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i+n,j+m)\cdot \mathbf{K}(n,m)$, where $\mathbf{O}$ has dimensions $(x-2, y-2)$

 This is the case for a single channel image. If the input has several output channels, the kernel should have the same number of channels. For example for $c$ input channels, $I$ is now $(x,y,c)$ and $K$ is now $(k_x,k_y,c)$. The output becomes:

  $\mathbf{O}(i,j) = \sum_{l=0}^{c}\sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i+n,j+m,l)\cdot \mathbf{K}(n,m,l)$, where $\mathbf{O}$ has dimensions $(x-2, y-2)$

  Finally, each convolutional layer may have more than one kernel, causing the output to have several channels. For $p$ kernels, $K$ is now $(k_x, k_y, c, p)$ the output of the convolutional layers operation will be:

  $\mathbf{O}(i,j,h) = \sum_{l=0}^{c}\sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i+n,j+m,l)\cdot \mathbf{K}(n,m,l,h)$, where $\mathbf{O}$ has dimensions $(x-2, y-2, p)$.
