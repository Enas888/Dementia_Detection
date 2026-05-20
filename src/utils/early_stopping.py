# src/utils/early_stopping.py

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_theta = None

    def step(self, loss, theta):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_theta = theta.copy()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience