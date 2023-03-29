import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.leaky_ReLu import LeakyReLu
from ANN.activation_functions.linear import Linear
from ANN.activation_functions.reLu import ReLu
from ANN.activation_functions.sigmoid import Sigmoid
from ANN.activation_functions.softmax import Softmax
from ANN.activation_functions.tanh import TanH

rnd = np.random.default_rng()

activation_pairs = [
    (ReLu, tf.keras.activations.relu),
    (Softmax, tf.keras.activations.softmax),
    (TanH, tf.keras.activations.tanh),
    (Sigmoid, tf.keras.activations.sigmoid),
    (Linear, tf.keras.activations.linear),
]


def test_forward():
    for ann_activation, tf_activation in activation_pairs:
        test_inputs = rnd.standard_normal((10, 100))
        ann_forward = ann_activation().forward(cp.array(test_inputs))
        tf_forward = tf_activation(tf.convert_to_tensor(test_inputs))
        assert np.allclose(ann_forward, tf_forward)


def test_backward():
    for ann_activation, tf_activation in activation_pairs:
        test_inputs = rnd.standard_normal((10, 100))
        activation = ann_activation()
        _ = activation.forward(cp.array(test_inputs))
        ann_backward = activation.backward(cp.ones((1, 100)))
        tensor_inputs = tf.convert_to_tensor(test_inputs)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tensor_inputs)
            tf_forward = tf_activation(tensor_inputs)
        tf_backward = tape.gradient(tf_forward, tensor_inputs).numpy()
        assert np.allclose(tf_backward, ann_backward)
