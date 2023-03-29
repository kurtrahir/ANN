import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.reLu import ReLu
from ANN.layers.dense import Dense

rnd = np.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((100, 100))
    ann_layer = Dense(64, activation=ReLu(), input_shape=test_inputs.shape)
    weights = ann_layer.weights[:-1, :]
    bias = ann_layer.weights[-1, :]

    tf_layer = tf.keras.layers.Dense(64, activation="relu")
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    tf_layer.set_weights(weights=[weights.get(), bias.get()])
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    ann_forward = ann_layer.forward(test_inputs)
    assert np.allclose(tf_forward.numpy(), ann_forward, rtol=1e-6, atol=1e-6)


def test_backward():
    test_inputs = rnd.standard_normal((100, 100))
    ann_layer = Dense(64, activation=ReLu(), input_shape=test_inputs.shape)
    weights = ann_layer.weights[:-1, :]
    bias = ann_layer.weights[-1, :]

    tf_layer = tf.keras.layers.Dense(64, activation="relu")
    test_tensor = tf.convert_to_tensor(test_inputs)
    tf_forward = tf_layer(test_tensor)
    tf_layer.set_weights(weights=[weights.get(), bias.get()])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(test_tensor)
        tf_forward = tf_layer(test_tensor)
    tf_w_grad = tape.gradient(
        tf_forward,
        tf_layer.trainable_variables,
    )
    tf_x_grad = tape.gradient(tf_forward, test_tensor)

    print(tf_x_grad)

    ann_forward = ann_layer.forward(test_inputs)
    ann_backward = ann_layer.backward(cp.ones((1, 64)))

    assert np.allclose(tf_w_grad[1], ann_layer.d_weights[-1], rtol=1e-5, atol=1e-5)
    assert np.allclose(tf_w_grad[0], ann_layer.d_weights[:-1], rtol=1e-5, atol=1e-5)
    assert np.allclose(tf_x_grad, ann_backward, rtol=1e-5, atol=1e-5)
