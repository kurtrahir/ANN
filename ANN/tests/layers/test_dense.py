import numpy as np
import tensorflow as tf

from ANN.activation_functions.linear import Linear
from ANN.layers.dense import Dense

rnd = np.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((100, 100))
    ann_layer = Dense(64, activation=Linear(), input_shape=test_inputs.shape)
    weights = ann_layer.weights[:-1, :]
    bias = ann_layer.weights[-1, :]

    tf_layer = tf.keras.layers.Dense(64, activation="linear")
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    tf_layer.set_weights(weights=[weights, bias])
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    ann_forward = ann_layer.forward(test_inputs)
    assert np.allclose(tf_forward.numpy(), ann_forward, rtol=1e-7, atol=1e-7)
