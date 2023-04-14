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
They typically consist of a certain number of neurons, who all receive the same input and learn a linear transformation which is then passed through an activation function. The activation function usually introduces some sort of non linearity, allowing the network to learn non-linear relationships. Given input $\mathbf{I} = [I_0, I_1, ..., I_n]$ a neuron with weights $\mathbf{w}= \left [ \begin{matrix}w_0 \\ w_1 \\ ... \\ w_n \end{matrix} \right]$, bias $b$ and activation function $\sigma(x)$ will compute $\sigma(\mathbf{w} \cdot I + b)$.

A densely connected layer with S neurons can be expressed using a weight matrix W consisting of S weight vectors $\mathbf{W} = [\mathbf{w_0},\mathbf{w_1},...,\mathbf{w_n}] = \left [\begin{matrix} w_{00} & w_{10} & ... & w_{S0} \\ w_{01} & w_{11} & ... & w_{S1} \\ ...& ...& ...& ... \\ w_{0n} & w_{1n} & ... & w_{Sn} \end{matrix}\right ]$ and a bias vector $\mathbf{b} = [b_0,b_1,...,b_S]$

This allows the activation of the dense layer to be expressed as: $a = \sigma(\mathbf{W\cdot I + b} )$.

A barebones example of a dense layer is therefore:

```python
class Dense(Layer):
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.inputs = None

    def forward(self, inputs):
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

This is extensible to a network with any number of layers $L$, simply backpropagate the gradient with regards to the input of a layer to the previous layer until the entire network has been traversed.

We know ${d\mathcal{z} \over d\mathbf{W}} = I$ and ${d\mathcal{z} \over d\mathbf{b}} = 1$ for all layers.

If we provide the gradient to be propagated to the backwards method, we can write it in a general purpose format (ignoring whether the layer in question is an output, hidden or input layer).

The barebones implementation for the dense layer gradient caculation  is therefore:

```python
def backward(self, gradient):
    d_activation = self.activation.backward(gradient)
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

#### __Structure and function__

With input $\mathbf{I}$ of size $(x,y)$ and kernel $\mathbf{K}$ of size $(k_x,k_y)$, the output $\mathbf{O}$ with dimensions $(x-k_x+1,y-k_y+1)$ of the convolutional layer at index $(i,j)$ is:

$$\mathbf{O}(i,j) = \sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i+n,j+m)\cdot \mathbf{K}(n,m)$$

This is technically the cross correlation ($\ast$) of the input and the kernel. These layers are called convolution for reasons that will become clear during the gradients discussion. This is the case for a single channel image. If the input has several output channels, the kernel should have the same number of channels. For example for $c$ input channels, $I$ is now $(x,y,c)$ and $K$ is now $(k_x,k_y,c)$. The output becomes:

$$\mathbf{O}(i,j) = \sum_{l=0}^{c}\sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i+n,j+m,l)\cdot \mathbf{K}(n,m,l)$$

Furthermore, each convolutional layer may have more than one kernel, causing the output to have several channels. For $p$ kernels, $K$ is now $(p, k_x, k_y, c)$ the output of the convolutional layers has dimensions $(x-k_x+1, y-k_y+1, p)$, and is:

$$\mathbf{O}(i,j,h) = \sum_{l=0}^{c}\sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i+n,j+m,l)\cdot \mathbf{K}(h,n,m,l)$$

To come back to the idea of parameters, if we consider the same example image, and instead of using 128 neurons, we learn 128 $(3\times 3)$ filters, we get $3\times 3 \times 3 \times 128 = 3456$ parameters, a much more manageable quantity.

One more consideration for the convolutional layer's oepration is that the kernel may be slid over the input in different manners by changing the step size. A standard operation has a step size of 1 in both spatial dimensions $(1,1)$ but larger step sizes can reduce the overlap of the receptive fields and achieve further dimensionality reduction of the input. Let $s_x, s_y$ be the step size in the $x$ and $y$ dimensions respectively, the output now has dimensions $({x-k_x \over s_x} + 1, {y-k_y \over s_y} + 1, p)$

$$\mathbf{O}(i,j,h) = \sum_{l=0}^{c}\sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i\times s_x+n,j\times s_y+m,l)\cdot \mathbf{K}(n,m,l,h)$$

Finally, same as for the dense layers, convolutional layers have an activation function and a bias vector $\mathbf{b}=[b_0,b_1,...,b_p]$. This means that the output of a convolutional layer is.

$$\mathbf{O}(i,j,h) = \sigma\left (\sum_{l=0}^{c}\sum_{n=0}^{k_x} \sum_{m=0}^{k_y} \mathbf{I}(i\times s_x+n,j\times s_y+m,l)\cdot \mathbf{K}(n,m,l,h) + b_h\right )$$

To compute this in an efficient manner, `cupy`'s `get_strided_view` was utilized. `cupy` arrays are stored in contiguous memory, and their structure is stored using strides. Typically, this means that adjacent elements in the first dimension are accessed by taking a stride of the size of the data type in being used, while accessing adjacent elements in the second dimension is achieved by taking a stride of the size of the data type multiplied by the size of the first dimension.

To avoid iterating over each image when passing it through a convolutional layer, we can exploit `stride_tricks` to create an array conducive to a single operation. Notably an array of size $({x-k_x \over s_x} + 1,{y-k_y \over s_y} + 1,p,k,k,c)$. The first three dimensions correspond to the output size of the operation, while the last three dimensions correspond to the size of an individual kernel.

Instead of creating an array of such size in memory, we change the strides to be taken to access the next element in each dimension. This allows us to have the same value exist in different indices of the array. Let $(m_x, m_y, m_z)$ be the stride size of the standard representation of the array in memory. To prepare for an efficient operation, we obtain a view with stride sizes: $(m_x \times s_z, m_y \times s_y, 0, m_x, m_y, m_z)$.
Notable features of these strides:
 - The first two dimensions stride in accordance with the step size of the operation.
 - The third dimension does not stride at all: this has the effect that the array is in effect repeated $p$ times in this dimension. This is done as all $p$ kernels operate on the same input data.
 - The last three dimensions are normal strides, as we want to access individual values.

By performing the dot product over the last three dimensions of the view and the last three dimensions of the kernel we are now left exactly with the desired output.

This is the barebones implementation of the strided dot product:

```python
import cupy as cp
def corr_multi_in_out(inputs, kernels, step_size):
    strided_input = cp.lib.stride_tricks.as_strided(
        inputs,
        shape=(
            (inputs.shape[0] - kernels.shape[0]) / step_size[0] + 1,
            (inputs.shape[1] - kernels.shape[1]) / step_size[1] + 1,
            kernels.shape[0],
            kernels.shape[1],
            kernels.shape[2],
            kernels.shape[3]
        ),
        strides=(
            inputs.strides[0] * step_size[0],
            inputs.strides[1] * step_size[1],
            0,
            inputs.strides[0],
            inputs.strides[1],
            inputs.strides[2],
        )
    )
    return cp.einsum(
        'ijklmn,klmn->ijk',
        strided_input,
        kernels
    )
```

And this is therefore the barebones implementation of a convolutional layer:

```python
class Conv2D(Layer):
    def __init__(self, weights, bias, activation, step_size):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.step_size = step_size
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation.forward(
            corr_multi_in_out(
                inputs, self.weights, self.step_size
            ) + self.bias
        )
```

#### __Gradients__

The funcamental idea of backpropagation applies to convolutional paramaters. To figure out the gradient with regards to the output, one needs to identify all the cells which were affected by each parameter.

Consider a $(3\times 3)$ input image $I$ and a $(2\times2)$ kernel $K$, being correlated with a $(1,1)$ step size, with an added bias $b$.
$$
\mathbf{I} = \left[
    \begin{matrix}
        x_{00} & x_{01} & x_{02} \\
        x_{10} & x_{11} & x_{12} \\
        x_{20} & x_{21} & x_{22} \\
    \end{matrix}
\right ]
\mathbf{K} =
\left[
    \begin{matrix}
        k_{00} & k_{01} \\
        k_{10} & k_{11} \\
    \end{matrix}
\right ]
$$
We will have a $(2\times2)$ output image $\mathbf{Z}$:
$$
\mathbf{Z} = \left[
    \begin{matrix}
        z_{00} & z_{01} \\
        z_{10} & z_{11} \\
    \end{matrix}
\right ]
$$

The following is true:

$$
    z_{00} = x_{00}k_{00} + x_{01}k_{01} + x_{10}k_{10} + x_{11}k_{11} + b\newline
    z_{01} = x_{01}k_{00} + x_{02}k_{01} + x_{11}k_{10} + x_{12}k_{11} + b\newline
    z_{10} = x_{10}k_{00} + x_{11}k_{01} + x_{20}k_{10} + x_{21}k_{11} + b\newline
    z_{11} = x_{11}k_{00} + x_{12}k_{01} + x_{21}k_{10} + x_{22}k_{11} + b\newline
$$

Let's look at ${d\mathbf{Z} \over d\mathbf{K}}$. We have

$$
    {d\mathbf{Z} \over d\mathbf{K}} =
    \left [
        \begin{matrix}
            {dZ \over dk_{00}} & {dZ \over dk_{01}} \\
            {dZ \over dk_{10}} & {dZ \over dk_{11}} \\
        \end{matrix}
    \right ]
$$
Where
$$
    {d\mathbf{Z} \over dk_{00}} =
    \left [
        \begin{matrix}
            {dz_{00} \over dk_{00}} & {dz_{01} \over dk_{00}} \\
            {dz_{10} \over dk_{00}} & {dz_{11} \over dk_{00}} \\
        \end{matrix}
    \right ] =
    \left [
        \begin{matrix}
            x_{00} & x_{01} \\
            x_{10} & x_{11} \\
        \end{matrix}
    \right ]
$$

And similarly:

$$
    {d\mathbf{Z} \over dk_{01}} =
    \left [
        \begin{matrix}
            x_{01} & x_{02} \\
            x_{11} & x_{12} \\
        \end{matrix}
    \right ]\text{, }{d\mathbf{Z} \over dk_{10}} =
    \left [
        \begin{matrix}
            x_{10} & x_{11} \\
            x_{20} & x_{21} \\
        \end{matrix}
    \right ] \text{, } {d\mathbf{Z} \over dk_{11}} =
    \left [
        \begin{matrix}
            x_{11} & x_{12} \\
            x_{21} & x_{22} \\
        \end{matrix}
    \right ]
$$.

The derivative of the output $\mathbf{O}$ with regards to $Z$ will have the shape of the output $(2\times 2)$, and the total influence of each individual kernel component can be obtained by summing over the gradient matrix. And so we see that to obtain the loss gradients for $\mathbf{K}$, it is a correlation operation:

$$
    {d\mathcal{L} \over d\mathbf{K}} = {d\mathcal{L} \over d\mathbf{Z}} \cdot {d\mathbf{Z} \over d\mathbf{K}} = \mathbf{I} \ast {d\mathcal{L} \over d\mathbf{Z}}
$$

When lookin at the bias we see that ${d\mathbf{Z} \over db} = \left [ \begin{matrix} 1 & 1 \\ 1 & 1 \\ \end{matrix} \right ]$ and so:

$$
    {d\mathcal{L} \over db} = {d\mathcal{L} \over d\mathbf{Z}} \cdot  {d\mathbf{Z} \over db} = {d\mathcal{L} \over d\mathbf{Z}}
$$

And finally, to be able to backpropagate to eventual earlier convolutional layers, let's look at ${d\mathbf{Z} \over d\mathbf{I}}$.
We have:
$$
    {d\mathbf{Z} \over d\mathbf{I}} =
    \left [
        \begin{matrix}
            {d\mathbf{Z} \over dx_{00}} & {d\mathbf{Z} \over dx_{01}} & {d\mathbf{Z} \over dx_{02}} \\
            {d\mathbf{Z} \over dx_{10}} & {d\mathbf{Z} \over dx_{11}} & {d\mathbf{Z} \over dx_{12}} \\
            {d\mathbf{Z} \over dx_{20}} & {d\mathbf{Z} \over dx_{21}} & {d\mathbf{Z} \over dx_{22}} \\
        \end{matrix}
    \right ]
$$

Where the corner components of the image are each only affected by a single kernel term:
$$
    {d\mathbf{Z} \over dx_{00}} = k_{00} \text{, }
    {d\mathbf{Z} \over dx_{02}} = k_{01} \text{, }
    {d\mathbf{Z} \over dx_{20}} = k_{10} \text{, }
    {d\mathbf{Z} \over dx_{22}} = k_{11}
$$
The middle components of the edges are affected by two kernel terms:
$$
    {d\mathbf{Z} \over dx_{01}} = k_{01} + k_{00}\text{, }
    {d\mathbf{Z} \over dx_{10}} = k_{10} + k_{00} \text{, }
    {d\mathbf{Z} \over dx_{12}} = k_{11} + k_{01} \text{, }
    {d\mathbf{Z} \over dx_{21}} = k_{11} + k_{10}
$$
And the center component is affected by all kernel terms:
$$
    {d\mathbf{Z} \over dx_{11}} = k_{00} + k_{01} + k_{10} + k_{11}
$$

To obtain this effect, we rotate the kernel and pad ${d\mathcal{L} \over d\mathbf{Z}}$
