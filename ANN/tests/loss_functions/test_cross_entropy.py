import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.softmax import Softmax
from ANN.loss_functions.cross_entropy import CrossEntropy

rnd = np.random.default_rng()


def test_forward():
    N_CLASSES = 100
    N_SAMPLES = 100
    test_inputs = Softmax().forward(
        cp.array(rnd.standard_normal((N_SAMPLES, N_CLASSES)))
    )
    labels = rnd.integers(0, N_CLASSES, (N_SAMPLES, 1))
    test_labels = np.zeros((N_SAMPLES, N_CLASSES))
    for i in range(N_SAMPLES):
        test_labels[i, labels[i]] = 1
    test_labels = test_labels
    cross_entropy = CrossEntropy()

    ann_forward = cross_entropy.forward(test_inputs, cp.array(test_labels))

    tf_forward = tf.keras.losses.categorical_crossentropy(
        test_labels, cp.asnumpy(test_inputs)
    )

    print(ann_forward.shape)
    print(tf_forward.shape)

    assert np.allclose(cp.asnumpy(ann_forward), tf_forward.numpy())


def test_backward():
    N_CLASSES = 100
    N_SAMPLES = 100
    test_inputs = Softmax().forward(
        cp.array(rnd.standard_normal((N_SAMPLES, N_CLASSES)))
    )
    labels = rnd.integers(0, N_CLASSES, (N_SAMPLES, 1))
    test_labels = np.zeros((N_SAMPLES, N_CLASSES))
    for i in range(N_SAMPLES):
        test_labels[i, labels[i]] = 1
    test_labels = test_labels
    cross_entropy = CrossEntropy()

    ann_forward = cross_entropy.forward(test_inputs, cp.array(test_labels))
    ann_backward = cross_entropy.backward(test_inputs, cp.array(test_labels))
    test_tensor = tf.convert_to_tensor(cp.asnumpy(test_inputs))

    with tf.GradientTape() as tape:
        tape.watch(test_tensor)
        tf_forward = tf.keras.losses.categorical_crossentropy(
            test_labels, test_tensor, from_logits=False
        )
    tf_gradient = tape.gradient(tf_forward, test_tensor)

    print(tf_gradient.shape)
    print(ann_backward.shape)

    assert np.allclose(cp.asnumpy(ann_backward), tf_gradient.numpy())
