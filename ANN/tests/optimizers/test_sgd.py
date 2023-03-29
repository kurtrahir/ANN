import cupy as cp
import numpy as np
import tensorflow as tf

from ANN.activation_functions.linear import Linear
from ANN.activation_functions.reLu import ReLu
from ANN.activation_functions.sigmoid import Sigmoid
from ANN.activation_functions.softmax import Softmax
from ANN.activation_functions.tanh import TanH
from ANN.layers.convolutional import Conv2D
from ANN.layers.dense import Dense
from ANN.layers.max_pooling import MaxPool2D
from ANN.loss_functions.cross_entropy import CrossEntropy
from ANN.loss_functions.mean_square_error import MSE
from ANN.models.sequential import Sequential
from ANN.optimizers.sgd import SGD

rnd = np.random.default_rng()


def test_dense_sgd():
    ann_losses = [MSE()]
    tf_losses = [tf.keras.losses.mse]

    ann_activations = [Linear(), Sigmoid(), TanH(), Softmax()]
    tf_activations = ["linear", "sigmoid", "tanh", "softmax"]

    for ann_loss, tf_loss in zip(ann_losses, tf_losses):
        for ann_activation, tf_activation in zip(ann_activations, tf_activations):
            print(ann_loss, tf_loss)
            print(ann_activation, tf_activation)

            N_SAMPLES = 2
            N_FEATURES = 20
            LEARNING_RATE = 1
            test_inputs = rnd.standard_normal((N_SAMPLES, N_FEATURES))
            test_labels = np.log(np.abs(np.sum(test_inputs, axis=-1, keepdims=True)))
            optimizer = SGD(learning_rate=LEARNING_RATE, loss=ann_loss)

            ann_model = Sequential(
                layers=[
                    Dense(3, activation=ann_activation),
                    Dense(3, activation=ann_activation),
                ],
                optimizer=optimizer,
            )

            _ = ann_model.forward(cp.array(test_inputs))

            weights = []
            biases = []

            for layer in ann_model.layers:
                weights.append(cp.asnumpy(layer.weights[:-1]))
                biases.append(cp.asnumpy(layer.weights[-1]))

            test_tensor = tf.convert_to_tensor(test_inputs)

            tf_model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(3, activation=tf_activation),
                    tf.keras.layers.Dense(3, activation=tf_activation),
                ]
            )

            tf_model.compile(
                optimizer=tf.keras.optimizers.SGD(LEARNING_RATE), loss=tf_loss
            )
            tf_model(test_tensor)

            for ann_layer, tf_layer in zip(ann_model.layers, tf_model.layers):
                tf_layer.set_weights(
                    [
                        cp.asnumpy(ann_layer.weights[:-1]),
                        cp.asnumpy(ann_layer.weights[-1]),
                    ]
                )

            for i, (ann_layer, tf_layer) in enumerate(
                zip(ann_model.layers, tf_model.layers)
            ):
                tf_weights, tf_biases = tf_layer.get_weights()
                assert np.allclose(
                    ann_layer.weights[:-1], tf_weights, rtol=1e-6, atol=1e-6
                )
                assert np.allclose(
                    ann_layer.weights[-1], tf_biases, rtol=1e-4, atol=1e-4
                )

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(test_tensor)
                tf_forward = tf_model(test_tensor)

            tf_w_grad = tape.gradient(
                tf_forward,
                tf_model.trainable_variables,
            )
            tf_x_grad = tape.gradient(tf_forward, test_tensor)

            tf_model.fit(test_tensor, test_labels, epochs=1, batch_size=N_SAMPLES)
            ann_model.train(
                cp.array(test_inputs),
                cp.array(test_labels),
                epochs=1,
                batch_size=N_SAMPLES,
            )

            for i, (ann_layer, tf_layer) in enumerate(
                zip(ann_model.layers, tf_model.layers)
            ):
                tf_weights, tf_biases = tf_layer.get_weights()
                assert not np.allclose(weights[i], ann_layer.weights[:-1])
                print(tf_w_grad[i * 2].shape)
                print(weights[i].shape)
                assert np.allclose(
                    cp.asnumpy(ann_layer.weights[-1]), tf_biases, rtol=1e-4, atol=1e-4
                )
                assert np.allclose(
                    cp.asnumpy(ann_layer.weights[:-1]), tf_weights, rtol=1e-4, atol=1e-4
                )


def test_loss_propagation_mse():
    N_SAMPLES = 1
    N_FEATURES = 2
    N_NEURONS = 2
    ann_loss = MSE()
    tf_loss = tf.keras.losses.mse

    ann_activations = [Linear(), ReLu(), Sigmoid(), TanH(), Softmax()]
    tf_activations = ["linear", "relu", "sigmoid", "tanh", "softmax"]

    for ann_activation, tf_activation in zip(ann_activations, tf_activations):
        print(tf_activation, tf_loss)
        ann_layer = Dense(
            N_NEURONS, activation=ann_activation, input_shape=(N_SAMPLES, N_FEATURES)
        )
        tf_layer = tf.keras.layers.Dense(N_NEURONS, activation=tf_activation)

        sample_input = rnd.standard_normal((N_SAMPLES, N_FEATURES))
        sample_label = rnd.standard_normal((N_SAMPLES, N_NEURONS))

        ann_input = cp.array(sample_input)
        ann_label = cp.array(sample_label)

        tf_input = tf.convert_to_tensor(sample_input)
        tf_label = tf.convert_to_tensor(sample_label)

        _ = tf_layer(tf_input)

        tf_layer.set_weights(
            [cp.asnumpy(ann_layer.weights[:-1]), cp.asnumpy(ann_layer.weights[-1])]
        )

        ann_d_loss = ann_loss.backward(ann_layer.forward(sample_input), ann_label)
        ann_d_x = ann_layer.backward(ann_d_loss)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_input)
            tf_pred = tf_layer(tf_input)
            loss = tf_loss(tf_label, tf_pred)
        tf_d_loss = tape.gradient(loss, tf_pred)
        tf_d_weights = tape.gradient(loss, tf_layer.trainable_variables)
        tf_d_x = tape.gradient(loss, tf_input)
        assert np.allclose(tf_d_loss.numpy(), cp.asnumpy(ann_d_loss))
        assert np.allclose(
            tf_d_weights[0].numpy(), cp.asnumpy(ann_layer.d_weights[:-1])
        )
        assert np.allclose(tf_d_weights[1].numpy(), cp.asnumpy(ann_layer.d_weights[-1]))
        assert np.allclose(tf_d_x.numpy(), cp.asnumpy(ann_d_x))


def test_loss_propagation_cross_entropy():
    N_SAMPLES = 1
    N_FEATURES = 2
    N_NEURONS = 2
    ann_loss = CrossEntropy()
    tf_loss = tf.keras.losses.categorical_crossentropy

    ann_activations = [Softmax()]
    tf_activations = ["softmax"]

    for ann_activation, tf_activation in zip(ann_activations, tf_activations):
        print(tf_activation, tf_loss)
        ann_layer = Dense(
            N_NEURONS, activation=ann_activation, input_shape=(N_SAMPLES, N_FEATURES)
        )
        tf_layer = tf.keras.layers.Dense(N_NEURONS, activation=tf_activation)

        sample_input = rnd.standard_normal((N_SAMPLES, N_FEATURES))
        classes = rnd.integers(0, N_NEURONS, (N_SAMPLES, 1))
        sample_label = np.zeros((N_SAMPLES, N_NEURONS), dtype=np.float32)
        for i in range(N_SAMPLES):
            sample_label[i, classes[i]] = 1

        ann_input = cp.array(sample_input)
        ann_label = cp.array(sample_label)

        tf_input = tf.convert_to_tensor(sample_input)
        tf_label = tf.convert_to_tensor(sample_label)

        _ = tf_layer(tf_input)

        tf_layer.set_weights(
            [cp.asnumpy(ann_layer.weights[:-1]), cp.asnumpy(ann_layer.weights[-1])]
        )

        ann_d_loss = ann_loss.backward(ann_layer.forward(sample_input), ann_label)
        ann_d_x = ann_layer.backward(ann_d_loss)

        tf_loss_object = tf.keras.losses.CategoricalCrossentropy()

        tf_pred = tf.convert_to_tensor(tf_layer(tf_input).numpy())

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_pred)
            loss = tf_loss_object(tf_label, tf_pred)

        tf_d_loss = tape.gradient(loss, tf_pred)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_input)
            tf_pred = tf_layer(tf_input)
            loss = tf_loss_object(tf_label, tf_pred)

        tf_d_weights = tape.gradient(loss, tf_layer.trainable_variables)
        tf_d_x = tape.gradient(loss, tf_input)

        assert np.allclose(tf_d_loss.numpy(), cp.asnumpy(ann_d_loss))
        print(tf_d_weights[0].numpy() - cp.asnumpy(ann_layer.d_weights[:-1]))
        assert np.allclose(
            tf_d_weights[0].numpy(), cp.asnumpy(ann_layer.d_weights[:-1])
        )
        assert np.allclose(tf_d_weights[1].numpy(), cp.asnumpy(ann_layer.d_weights[-1]))
        assert np.allclose(tf_d_x.numpy(), cp.asnumpy(ann_d_x))
