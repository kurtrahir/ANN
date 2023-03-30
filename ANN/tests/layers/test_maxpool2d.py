import cupy as cp
import numpy as cp
import tensorflow as tf

from ANN.layers.max_pooling import MaxPool2D

rnd = cp.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((100, 28, 28, 1))
    ann_layer = MaxPool2D(kernel_size=(2, 2), step_size=(2, 2), padding="valid")

    tf_layer = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )

    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs))
    ann_forward = ann_layer.forward(cp.array(test_inputs))
    assert cp.allclose(tf_forward.numpy(), ann_forward, rtol=1e-6, atol=1e-6)


def test_backward():
    test_inputs = rnd.standard_normal((3, 10, 10, 3))

    ann_layer = MaxPool2D(kernel_size=(2, 2), step_size=(2, 2), padding="valid")

    tf_layer = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )

    test_tensor = tf.convert_to_tensor(test_inputs)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(test_tensor)
        tf_forward = tf_layer(test_tensor)

    tf_x_grad = tape.gradient(tf_forward, test_tensor)
    ann_forward = ann_layer.forward(cp.array(test_inputs))
    ann_backward = ann_layer.backward(cp.ones((1, 1, 1, 1)))

    print(tf_x_grad)
    print(ann_backward)

    assert cp.allclose(tf_x_grad, ann_backward, rtol=1e-6, atol=1e-6)
