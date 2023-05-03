import cupy as cp

import ANN


def cnn_model_test(model):
    rnd = cp.random.default_rng()

    N_SAMPLES = 100

    inputs = rnd.standard_normal((N_SAMPLES, 28, 28, 1))
    targets = rnd.integers(0, 10, N_SAMPLES)

    def one_hot_encode(labels, n_labels):
        new_labels = cp.zeros((labels.shape[0], n_labels))
        for i in range(labels.shape[0]):
            new_labels[i, labels[i]] = 1
        return new_labels

    targets = one_hot_encode(targets, 10)

    model.train(inputs, targets, epochs=10, batch_size=100)

    assert isinstance(model.forward(inputs), cp.ndarray)
    assert isinstance(model.forward(inputs[0:1]), cp.ndarray)


def test_cnn():
    for padding in ["full", "valid", "same"]:
        model = ANN.models.Sequential(
            [
                ANN.layers.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.activation_functions.ReLu(),
                ),
                ANN.layers.Flatten(),
                ANN.layers.Dense(
                    n_neurons=10, activation=ANN.activation_functions.Softmax()
                ),
            ],
            ANN.optimizers.SGD(
                loss=ANN.loss_functions.CrossEntropy(), learning_rate=1e-3
            ),
        )
        cnn_model_test(model)


def test_cnn_deep():
    for padding in ["full", "valid", "same"]:
        model = ANN.models.Sequential(
            [
                ANN.layers.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.activation_functions.ReLu(),
                ),
                ANN.layers.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.activation_functions.ReLu(),
                ),
                ANN.layers.Flatten(),
                ANN.layers.Dense(
                    n_neurons=10, activation=ANN.activation_functions.Softmax()
                ),
            ],
            ANN.optimizers.SGD(
                loss=ANN.loss_functions.CrossEntropy(), learning_rate=1e-3
            ),
        )
        cnn_model_test(model)


def test_cnn_deep_maxpool():
    for padding in ["full", "valid", "same"]:
        model = ANN.models.Sequential(
            [
                ANN.layers.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.activation_functions.ReLu(),
                ),
                ANN.layers.MaxPool2D(
                    kernel_size=(2, 2), step_size=(2, 2), padding="valid"
                ),
                ANN.layers.Conv2D(
                    n_filters=2,
                    kernel_shape=(2, 2),
                    step_size=(1, 1),
                    padding=padding,
                    activation_function=ANN.activation_functions.ReLu(),
                ),
                ANN.layers.MaxPool2D(
                    kernel_size=(2, 2), step_size=(2, 2), padding="valid"
                ),
                ANN.layers.Flatten(),
                ANN.layers.Dense(
                    n_neurons=10, activation=ANN.activation_functions.Softmax()
                ),
            ],
            ANN.optimizers.SGD(
                loss=ANN.loss_functions.CrossEntropy(), learning_rate=1e-3
            ),
        )
        cnn_model_test(model)
