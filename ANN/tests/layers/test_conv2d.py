import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.linear import Linear
from ANN.layers.convolutional import Conv2D

rnd = np.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((100, 9, 9, 1))
    ann_layer = Conv2D(
        n_filters=64,
        kernel_shape=(3, 3),
        activation_function=Linear(),
        input_shape=test_inputs.shape,
        padding="valid",
    )
    weights = np.moveaxis(cp.asnumpy(ann_layer.weights), 0, 3)
    bias = cp.asnumpy(ann_layer.bias)

    tf_layer = tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), activation="linear", padding="valid"
    )
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    tf_layer.set_weights(weights=[weights, bias])
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    ann_forward = ann_layer.forward(cp.array(test_inputs))
    assert np.allclose(tf_forward.numpy(), ann_forward, rtol=1e-6, atol=1e-6)


def test_backward():
    test_inputs = rnd.standard_normal((2, 3, 3, 3))
    ann_layer = Conv2D(
        n_filters=4,
        kernel_shape=(3, 3),
        activation_function=Linear(),
        input_shape=test_inputs.shape,
        padding="valid",
    )
    weights = np.moveaxis(cp.asnumpy(ann_layer.weights), 0, 3)
    bias = cp.asnumpy(ann_layer.bias)

    tf_layer = tf.keras.layers.Conv2D(
        4, kernel_size=(3, 3), activation="linear", padding="valid"
    )
    test_tensor = tf.convert_to_tensor(test_inputs)
    tf_forward = tf_layer(test_tensor)
    tf_layer.set_weights(weights=[weights, bias])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(test_tensor)
        tf_forward = tf_layer(test_tensor)
    tf_w_grad = tape.gradient(
        tf_forward,
        tf_layer.trainable_variables,
    )
    tf_x_grad = tape.gradient(tf_forward, test_tensor)
    ann_forward = ann_layer.forward(test_inputs)
    ann_backward = ann_layer.backward(cp.ones((1, 1, 1, 1)))
    print(tf_w_grad[0])
    print(cp.moveaxis(ann_layer.d_weights, 0, 3))

    assert np.allclose(tf_x_grad, ann_backward, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        tf_w_grad[0], cp.moveaxis(ann_layer.d_weights, 0, 3), rtol=1e-6, atol=1e-6
    )
    assert np.allclose(tf_w_grad[1], ann_layer.d_bias, rtol=1e-6, atol=1e-6)
