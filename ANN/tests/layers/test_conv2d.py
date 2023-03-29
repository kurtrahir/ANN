import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.linear import Linear
from ANN.activation_functions.reLu import ReLu
from ANN.activation_functions.sigmoid import Sigmoid
from ANN.activation_functions.tanh import TanH
from ANN.layers.convolutional import Conv2D

rnd = np.random.default_rng()


def test_forward():
    for padding in ["valid", "same"]:
        test_inputs = rnd.standard_normal((1, 28, 28, 3))
        ann_layer = Conv2D(
            n_filters=64,
            kernel_shape=(3, 3),
            activation_function=Linear(),
            input_shape=test_inputs.shape,
            padding=padding,
        )
        weights = np.moveaxis(cp.asnumpy(ann_layer.weights), 0, 3)
        bias = cp.asnumpy(ann_layer.bias)

        tf_layer = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation="linear", padding=padding
        )
        tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
        tf_layer.set_weights(weights=[weights, bias])
        tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
        ann_forward = ann_layer.forward(cp.array(test_inputs))
        assert np.allclose(tf_forward.numpy(), ann_forward, rtol=1e-6, atol=1e-6)


def test_backward():
    for t_a, a_a in [
        ("linear", Linear()),
        ("relu", ReLu()),
        ("sigmoid", Sigmoid()),
        ("tanh", TanH()),
    ]:
        for padding in ["valid", "same"]:
            print(padding)
            test_inputs = rnd.standard_normal((8, 28, 28, 3))
            ann_layer = Conv2D(
                n_filters=4,
                kernel_shape=(3, 3),
                activation_function=a_a,
                input_shape=test_inputs.shape,
                padding=padding,
            )
            weights = np.moveaxis(cp.asnumpy(ann_layer.weights), 0, 3)
            bias = cp.asnumpy(ann_layer.bias)

            tf_layer = tf.keras.layers.Conv2D(
                4, kernel_size=(3, 3), activation=t_a, padding=padding
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
            ann_forward = ann_layer.forward(cp.array(test_inputs))
            ann_backward = ann_layer.backward(cp.ones((1, 1, 1, 1)))

            assert ann_forward.shape == tf_forward.shape
            assert np.allclose(
                cp.asnumpy(ann_forward), tf_forward.numpy(), rtol=1e-5, atol=1e-5
            )
            assert np.allclose(tf_x_grad, ann_backward, rtol=1e-5, atol=1e-5)
            temp_ann_weights = cp.moveaxis(ann_layer.d_weights, 0, 3)
            print(np.max(tf_w_grad[0] - cp.asnumpy(cp.around(temp_ann_weights, 5))))
            assert np.allclose(
                tf_w_grad[0],
                cp.moveaxis(ann_layer.d_weights, 0, 3),
                rtol=1e-5,
                atol=1e-5,
            )
            assert np.allclose(tf_w_grad[1], ann_layer.d_bias, rtol=1e-6, atol=1e-6)
