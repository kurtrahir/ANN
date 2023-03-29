import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.loss_functions.mean_square_error import MSE

rnd = np.random.default_rng()


def test_forward():
    N_CLASSES = 100
    N_SAMPLES = 100
    test_inputs = cp.array(rnd.standard_normal((N_SAMPLES, N_CLASSES)))
    labels = rnd.integers(0, N_CLASSES, (N_SAMPLES, 1))
    test_labels = np.zeros((N_SAMPLES, N_CLASSES))
    for i in range(N_SAMPLES):
        test_labels[i, labels[i]] = 1
    test_labels = test_labels
    mse = MSE()

    ann_forward = mse.forward(test_inputs, cp.array(test_labels))

    tf_forward = tf.keras.losses.mse(test_labels, cp.asnumpy(test_inputs))

    print(ann_forward.shape)
    print(tf_forward.shape)

    assert np.allclose(cp.asnumpy(ann_forward), tf_forward.numpy())


def test_backward():
    N_CLASSES = 100
    N_SAMPLES = 100
    test_inputs = cp.array(rnd.standard_normal((N_SAMPLES, N_CLASSES)))
    labels = rnd.integers(0, N_CLASSES, (N_SAMPLES, 1))
    test_labels = np.zeros((N_SAMPLES, N_CLASSES))
    for i in range(N_SAMPLES):
        test_labels[i, labels[i]] = 1
    test_labels = test_labels
    mse = MSE()

    ann_forward = mse.forward(test_inputs, cp.array(test_labels))
    ann_backward = mse.backward(test_inputs, cp.array(test_labels))
    test_tensor = tf.convert_to_tensor(cp.asnumpy(test_inputs))

    with tf.GradientTape() as tape:
        tape.watch(test_tensor)
        tf_forward = tf.keras.losses.mse(test_labels, test_tensor)
    tf_gradient = tape.gradient(tf_forward, test_tensor)

    tf_gradient = tf_gradient.numpy()
    ann_backward = cp.asnumpy(ann_backward)
    print(type(tf_gradient), type(ann_backward))
    print(tf_gradient - ann_backward)
    print((tf_gradient - ann_backward)[np.where((tf_gradient - ann_backward))])

    assert np.allclose(ann_backward, tf_gradient, atol=1e-3, rtol=1e-3)
