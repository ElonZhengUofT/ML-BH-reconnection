class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def __call__(self, current_loss):
        if self.best_val_loss is None:
            self.best_val_loss = current_loss
        elif self.best_val_loss - current_loss > self.min_delta:
            self.best_val_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            print(f"Early stopping counter {self.num_bad_epochs} of {self.patience}")
            if self.num_bad_epochs >= self.patience:
                print("Early stopping triggered!")
                self.should_stop = True
