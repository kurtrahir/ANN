import numpy as np
import tensorflow as tf

from ANN.activation_functions.leaky_ReLu import LeakyReLu

rnd = np.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((1, 100))
    ann_forward = LeakyReLu(0.3).forward(test_inputs)
    tf_forward = tf.keras.layers.LeakyReLU(0.3)(tf.convert_to_tensor(test_inputs))
    assert np.isclose(ann_forward, tf_forward).all()


def test_backward():
    test_inputs = rnd.standard_normal((1, 100))
    relu = LeakyReLu(0.3)
    _ = relu.forward(test_inputs)
    ann_backward = relu.backward(np.ones((1, 100)))
    tensor_inputs = tf.convert_to_tensor(test_inputs)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tensor_inputs)
        tf_forward = tf.keras.layers.LeakyReLU(0.3)(tensor_inputs)
    tf_backward = tape.gradient(tf_forward, tensor_inputs).numpy()
    assert np.isclose(tf_backward, ann_backward).all()
