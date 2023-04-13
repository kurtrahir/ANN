import numpy as np

from ANN.callbacks.callback import Callback


class ReduceLROnPlateau(Callback):
    def __init__(
        self, patience: int, factor: np.float32 = 2, loss_key: str = "validation_loss"
    ):
        """_summary_

        Args:
            patience (int): Number of epochs to wait before lr reduction
            factor (np.float32, optional): Factor by which to divide the learning rate. Defaults to 2.
            loss_key (str, optional): Loss to track. Defaults to "validation_loss".
        """
        self.patience = patience
        self.factor = factor
        self.loss_key = loss_key
        self.last_reduction = 0

    def call(self, model):
        """Checks whether the chosen loss metric at the most recent epoch is lower than
        it was "patience" epochs ago. If not, reduce learning rate by dividing
        it by factor

        Args:
            model (Model): Model to operate on
        """
        keys = list(model.history[self.loss_key].keys())[self.last_reduction :]
        if len(keys) < self.patience:
            return
        values = []
        for i in range(-self.patience, -1):
            values.append(model.history[self.loss_key][keys[i]])
        if model.history[self.loss_key][keys[-1]] < np.mean(np.array(values)):
            return
        model.optimizer.learning_rate /= self.factor
        self.last_reduction = keys[-1]
        print(f"Reduced learning rate to {model.optimizer.learning_rate}")
