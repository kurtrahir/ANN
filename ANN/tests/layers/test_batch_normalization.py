import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.layers.batch_normalization import BatchNormalization

rnd = cp.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((100, 28, 28, 1))
    ann_layer = BatchNormalization(epsilon=1e-3, momentum=0.9)

    tf_layer = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9)

    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs.get()))
    ann_forward = ann_layer.forward(test_inputs, False)
    print(tf_forward.numpy() - ann_forward.get())
    assert cp.allclose(tf_forward.numpy(), ann_forward, rtol=1e-6, atol=1e-6)


def test_backward():
    test_inputs = rnd.standard_normal((3, 10, 10, 3))

    ann_layer = BatchNormalization(epsilon=1e-3, momentum=0.9)

    tf_layer = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9)
    test_tensor = tf.convert_to_tensor(test_inputs.get(), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(test_tensor)
        tf_forward = tf_layer(test_tensor, True)

    tf_x_grad = tape.gradient(tf_forward, test_tensor).numpy()
    tf_w_grad = tape.gradient(tf_forward, tf_layer.trainable_variables)
    ann_forward = ann_layer.forward(test_inputs, True)
    ann_backward = ann_layer.backward(cp.ones((3, 10, 10, 3))).get()

    print(tf_x_grad.shape)
    print(ann_backward.shape)
    print(tf_layer.gamma.shape)
    print(ann_layer.weights.shape)
    print(tf_layer.beta.shape)
    print(ann_layer.bias.shape)
    print(tf_w_grad[0].shape)
    print(ann_layer.d_weights.shape)
    print(tf_w_grad[1].shape)
    print(ann_layer.d_bias.shape)
    assert cp.allclose(tf_x_grad, ann_backward, rtol=1e-6, atol=1e-6)
