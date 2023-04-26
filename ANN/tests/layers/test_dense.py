import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.linear import Linear
from ANN.activation_functions.reLu import ReLu
from ANN.activation_functions.sigmoid import Sigmoid
from ANN.activation_functions.softmax import Softmax
from ANN.activation_functions.tanh import TanH
from ANN.layers.dense import Dense
from ANN.loss_functions.cross_entropy import CrossEntropy
from ANN.loss_functions.mean_square_error import MSE

rnd = cp.random.default_rng()


def test_forward():
    test_inputs = rnd.standard_normal((100, 100)).astype(cp.float32)
    ann_layer = Dense(64, activation=ReLu(), input_shape=test_inputs.shape)
    weights = ann_layer.weights
    bias = ann_layer.bias

    tf_layer = tf.keras.layers.Dense(64, activation="relu")
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs.get()))
    tf_layer.set_weights(weights=[weights.get(), bias.get()])
    tf_forward = tf_layer(tf.convert_to_tensor(test_inputs.get())).numpy()
    ann_forward = ann_layer.forward(test_inputs, False).get()
    assert tf_forward.shape == ann_forward.shape
    assert cp.allclose(tf_forward, ann_forward, rtol=1e-6, atol=1e-6)


def test_backward():
    # test input generation
    test_inputs = rnd.standard_normal((100, 100), dtype=cp.float32)
    # Declare and initialize Dense layer
    ann_layer = Dense(64, activation=ReLu(), input_shape=test_inputs.shape)

    # Get weights and bias
    weights = ann_layer.weights
    bias = ann_layer.bias

    # Declare and initialize tf reference dense layer
    tf_layer = tf.keras.layers.Dense(64, activation="relu")
    test_tensor = tf.convert_to_tensor(test_inputs.get(), dtype=tf.float32)
    tf_forward = tf_layer(test_tensor)
    tf_layer.set_weights(weights=[weights.get(), bias.get()])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(test_tensor)
        tf_forward = tf_layer(test_tensor)
    # Get reference weight gradient
    tf_w_grad = tape.gradient(
        tf_forward,
        tf_layer.trainable_variables,
    )
    # Get reference input gradient
    tf_x_grad = tape.gradient(tf_forward, test_tensor)

    ann_forward = ann_layer.forward(test_inputs, training=True)
    ann_backward = ann_layer.backward(cp.ones((1, 64)))

    assert cp.allclose(tf_w_grad[1], ann_layer.d_bias, rtol=1e-5, atol=1e-5)
    assert cp.allclose(tf_w_grad[0], ann_layer.d_weights, rtol=1e-5, atol=1e-5)
    assert cp.allclose(tf_x_grad, ann_backward, rtol=1e-5, atol=1e-5)


def test_loss_propagation_mse():
    N_SAMPLES = 1
    N_FEATURES = 100
    N_NEURONS = 100
    ann_loss = MSE()
    tf_loss = tf.keras.losses.mse

    ann_activations = [Linear(), ReLu(), Sigmoid(), TanH(), Softmax()]
    tf_activations = ["linear", "relu", "sigmoid", "tanh", "softmax"]

    for ann_activation, tf_activation in zip(ann_activations, tf_activations):
        ann_layer = Dense(
            N_NEURONS, activation=ann_activation, input_shape=(N_SAMPLES, N_FEATURES)
        )
        tf_layer = tf.keras.layers.Dense(N_NEURONS, activation=tf_activation)

        sample_input = rnd.standard_normal((N_SAMPLES, N_FEATURES), dtype=cp.float32)
        sample_label = rnd.standard_normal((N_SAMPLES, N_NEURONS), dtype=cp.float32)

        ann_input = sample_input
        ann_label = sample_label

        tf_input = tf.convert_to_tensor(cp.asnumpy(sample_input), dtype=tf.float32)
        tf_label = tf.convert_to_tensor(cp.asnumpy(sample_label), dtype=tf.float32)

        _ = tf_layer(tf_input)

        tf_layer.set_weights(
            [cp.asnumpy(ann_layer.weights), cp.asnumpy(ann_layer.bias)]
        )

        ann_d_loss = ann_loss.backward(
            ann_layer.forward(ann_input, training=True), ann_label
        )
        ann_d_x = ann_layer.backward(ann_d_loss)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_input)
            tf_pred = tf_layer(tf_input)
            loss = tf_loss(tf_label, tf_pred)
        tf_d_loss = tape.gradient(loss, tf_pred)
        tf_d_weights = tape.gradient(loss, tf_layer.trainable_variables)
        tf_d_x = tape.gradient(loss, tf_input)
        assert cp.allclose(tf_d_loss.numpy(), ann_d_loss)
        assert cp.allclose(tf_d_weights[0].numpy(), ann_layer.d_weights)
        assert cp.allclose(tf_d_weights[1].numpy(), ann_layer.d_bias)
        assert cp.allclose(tf_d_x.numpy(), ann_d_x)


def test_loss_propagation_cross_entropy():
    N_SAMPLES = 1
    N_FEATURES = 10
    N_NEURONS = 2
    ann_loss = CrossEntropy()
    tf_loss = tf.keras.losses.categorical_crossentropy

    ann_activations = [Softmax()]
    tf_activations = ["softmax"]

    for ann_activation, tf_activation in zip(ann_activations, tf_activations):
        ann_layer = Dense(
            N_NEURONS, activation=ann_activation, input_shape=(N_SAMPLES, N_FEATURES)
        )
        tf_layer = tf.keras.layers.Dense(N_NEURONS, activation=tf_activation)

        sample_input = rnd.standard_normal((N_SAMPLES, N_FEATURES), dtype=cp.float32)
        classes = rnd.integers(0, N_NEURONS, (N_SAMPLES, 1))
        sample_label = cp.zeros((N_SAMPLES, N_NEURONS), dtype=cp.float32)
        for i in range(N_SAMPLES):
            sample_label[i, classes[i]] = 1

        ann_input = sample_input
        ann_label = sample_label

        tf_input = tf.convert_to_tensor(cp.asnumpy(sample_input), dtype=tf.float32)
        tf_label = tf.convert_to_tensor(cp.asnumpy(sample_label), dtype=tf.float32)

        _ = tf_layer(tf_input)

        tf_layer.set_weights(
            [cp.asnumpy(ann_layer.weights), cp.asnumpy(ann_layer.bias)]
        )

        ann_d_loss = cp.asnumpy(
            ann_loss.backward(ann_layer.forward(ann_input, training=True), ann_label)
        )
        ann_d_x = cp.asnumpy(ann_layer.backward(ann_d_loss))

        tf_pred = tf.convert_to_tensor(tf_layer(tf_input).numpy())

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_pred)
            loss = tf_loss(tf_label, tf_pred)

        tf_d_loss = tape.gradient(loss, tf_pred)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_input)
            tf_pred = tf_layer(tf_input)
            loss = tf_loss(tf_label, tf_pred)

        tf_d_weights = tape.gradient(loss, tf_layer.trainable_variables)
        tf_d_x = tape.gradient(loss, tf_input)

        assert np.allclose(tf_d_loss.numpy(), cp.asnumpy(ann_d_loss))
        assert np.allclose(tf_d_weights[1].numpy(), cp.asnumpy(ann_layer.d_bias))
        assert np.allclose(tf_d_weights[0].numpy(), cp.asnumpy(ann_layer.d_weights))
        assert np.allclose(tf_d_x.numpy(), cp.asnumpy(ann_d_x))
