import cupy as cp
import numpy as cp
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
from ANN.optimizers.adam import Adam
from ANN.optimizers.sgd import SGD

rnd = cp.random.default_rng()


def test_dense_adam():
    ann_losses = [MSE()]
    tf_losses = [tf.keras.losses.mse]

    ann_activations = [Linear, ReLu, Sigmoid, TanH, Softmax]
    tf_activations = ["linear", "relu", "sigmoid", "tanh", "softmax"]

    for ann_loss, tf_loss in zip(ann_losses, tf_losses):
        for ann_activation, tf_activation in zip(ann_activations, tf_activations):
            print(ann_loss, tf_loss)
            print(ann_activation, tf_activation)

            N_SAMPLES = 10
            N_FEATURES = 20
            LEARNING_RATE = 1
            BETA_1 = 0.5
            BETA_2 = 0.5
            EPSILON = 1e-7
            test_inputs = rnd.standard_normal((N_SAMPLES, N_FEATURES))
            test_labels = cp.log(cp.abs(cp.sum(test_inputs, axis=-1, keepdims=True)))
            optimizer = Adam(
                learning_rate=LEARNING_RATE,
                beta_1=BETA_1,
                beta_2=BETA_2,
                epsilon=EPSILON,
                loss=ann_loss,
            )

            ann_model = Sequential(
                layers=[
                    Dense(3, activation=ann_activation()),
                    Dense(3, activation=ann_activation()),
                ],
                optimizer=optimizer,
            )

            _ = ann_model.forward(cp.array(test_inputs))

            weights = []
            biases = []

            for layer in ann_model.layers:
                weights.append(cp.asnumpy(layer.weights))
                biases.append(cp.asnumpy(layer.bias))

            test_tensor = tf.convert_to_tensor(test_inputs)

            tf_model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(3, activation=tf_activation),
                    tf.keras.layers.Dense(3, activation=tf_activation),
                ]
            )

            tf_model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=LEARNING_RATE,
                    beta_1=BETA_1,
                    beta_2=BETA_2,
                    epsilon=EPSILON,
                ),
                loss=tf_loss,
            )
            tf_model(test_tensor)

            for ann_layer, tf_layer in zip(ann_model.layers, tf_model.layers):
                tf_layer.set_weights(
                    [
                        cp.asnumpy(ann_layer.weights),
                        cp.asnumpy(ann_layer.bias),
                    ]
                )

            for i, (ann_layer, tf_layer) in enumerate(
                zip(ann_model.layers, tf_model.layers)
            ):
                tf_weights, tf_biases = tf_layer.get_weights()
                assert cp.allclose(ann_layer.weights, tf_weights, rtol=1e-6, atol=1e-6)
                assert cp.allclose(ann_layer.bias, tf_biases, rtol=1e-4, atol=1e-4)

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
                print(i)
                tf_weights, tf_biases = tf_layer.get_weights()
                assert not cp.allclose(weights[i], ann_layer.weights)
                # Assert error less than .1%
                assert (
                    (ann_layer.weights.get() - tf_weights) / cp.linalg.norm(tf_weights)
                    < 0.001
                ).all()
                assert (
                    (ann_layer.bias.get() - tf_biases) / cp.linalg.norm(tf_biases)
                    < 0.001
                ).all()
