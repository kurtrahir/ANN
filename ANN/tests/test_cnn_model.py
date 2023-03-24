import numpy as np

import ANN


def cnn_model_test(model):
    rnd = np.random.default_rng()

    N_SAMPLES = 100

    inputs = rnd.standard_normal((N_SAMPLES, 28, 28, 1))
    targets = rnd.integers(0, 10, N_SAMPLES)

    def one_hot_encode(labels, n_labels):
        new_labels = np.zeros((labels.shape[0], n_labels))
        for i in range(labels.shape[0]):
            new_labels[i, labels[i]] = 1
        return new_labels

    targets = one_hot_encode(targets, 10)

    model.train(inputs, targets, epochs=10, batch_size=100)

    assert isinstance(model.forward(inputs), np.ndarray)
    assert isinstance(model.forward(inputs[0:1]), np.ndarray)


def test_cnn():
    for padding in ["full", "valid", "same"]:
        model = ANN.Sequential(
            [
                ANN.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.ReLu(),
                ),
                ANN.Flatten(),
                ANN.Dense(n_neurons=10, activation=ANN.Linear()),
            ],
            ANN.SGD(loss=ANN.CategoricalCrossEntropy(), learning_rate=1e-3),
        )
        cnn_model_test(model)


def test_cnn_deep():
    for padding in ["full", "valid", "same"]:
        model = ANN.Sequential(
            [
                ANN.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.ReLu(),
                ),
                ANN.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.ReLu(),
                ),
                ANN.Flatten(),
                ANN.Dense(n_neurons=10, activation=ANN.Linear()),
            ],
            ANN.SGD(loss=ANN.CategoricalCrossEntropy(), learning_rate=1e-3),
        )
        cnn_model_test(model)


def test_cnn_deep_maxpool():
    for padding in ["full", "valid", "same"]:
        model = ANN.Sequential(
            [
                ANN.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.ReLu(),
                ),
                ANN.MaxPool2D(kernel_size=(2, 2), step_size=(2, 2)),
                ANN.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.ReLu(),
                ),
                ANN.MaxPool2D(kernel_size=(2, 2), step_size=(2, 2)),
                ANN.Flatten(),
                ANN.Dense(n_neurons=10, activation=ANN.Linear()),
            ],
            ANN.SGD(loss=ANN.CategoricalCrossEntropy(), learning_rate=1e-3),
        )
        cnn_model_test(model)
