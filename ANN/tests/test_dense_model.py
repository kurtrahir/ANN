import cupy as np

import ANN


def dense_model_test(model):
    rnd = np.random.default_rng()

    N_SAMPLES = 100

    inputs = rnd.standard_normal((N_SAMPLES, 28 * 28))
    targets = rnd.integers(0, 10, N_SAMPLES)

    def one_hot_encode(labels, n_labels):
        new_labels = np.zeros((labels.shape[0], n_labels))
        for i in range(labels.shape[0]):
            new_labels[i, labels[i]] = 1
        return new_labels

    targets = one_hot_encode(targets, 10)

    model.train(inputs, targets, epochs=10, batch_size=100)

    assert type(model.forward(inputs)) is np.ndarray
    assert type(model.forward(inputs[0:1])) is np.ndarray


def test_dense():
    model = ANN.Sequential(
        [ANN.Dense(n_neurons=10, activation=ANN.Sigmoid())],
        ANN.SGD(loss=ANN.BinaryCrossEntropy(), learning_rate=1e-3),
    )
    dense_model_test(model)


def test_dense_deep():
    model = ANN.Sequential(
        [
            ANN.Dense(n_neurons=10, activation=ANN.ReLu()),
            ANN.Dense(n_neurons=10, activation=ANN.ReLu()),
            ANN.Dense(n_neurons=10, activation=ANN.ReLu()),
            ANN.Dense(n_neurons=10, activation=ANN.Sigmoid()),
        ],
        ANN.SGD(loss=ANN.BinaryCrossEntropy(), learning_rate=1e-3),
    )
    dense_model_test(model)
