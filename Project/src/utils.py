import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def split_date(dates, cutoff):
    train_date = [x for x in dates if x <= cutoff]
    test_date = sorted(list(set(dates) - set(train_date)))
    return train_date, test_date

class TimeSeriesDataset:
    def __init__(self, data, lookback, lookahead, stride=1, batch_size=32):
        self.data = data
        self.lookback = lookback
        self.lookahead = lookahead
        self.stride = stride
        self.batch_size = batch_size

    def to_dataloader(self, shuffle=False):
        """
        Convert the data to a PyTorch times series dataset and return a DataLoader.
        """
        input, output = self.to_timeseries_input(self.data)
        dataset = TensorDataset(input, output)
        data_loader = DataLoader(dataset, 
                                                    batch_size=self.batch_size, 
                                                    shuffle=shuffle)
        return data_loader

    def rolling_window(self, series, window_size):
        """
        Given a sequence, return a numpy array of the rolling window.
        """
        arr = []
        for i in range(0, series.shape[0] - window_size + 1, self.stride):
            arr.append(series[i : (i + window_size)])
        return np.array(arr)

    def to_timeseries_input(self, series):
        """
        Given a sequence, return a PyTorch tensor couple (inputs, outputs) 
        of the rolling window.
        """
        inputs = self.rolling_window(series[:-self.lookahead], self.lookback)
        outputs = self.rolling_window(series[self.lookback:], self.lookahead)
        
        return torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1), \
                 torch.tensor(outputs, dtype=torch.float32).unsqueeze(-1)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, 
                    optimizer, scheduler, device) -> None:
        """
        A utility class for training, validating and testing a PyTorch model.

        Attributes:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): The DataLoader for the training data.
            val_loader (DataLoader): The DataLoader for the validation data.
            test_loader (DataLoader): The DataLoader for the test data.
            criterion (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer.
            scheduler (Scheduler): The scheduler.
            device (torch.device): The device to train on (e.g., 'cpu' or 'cuda' or 'mps').

        Methods:
            train(epochs, patience): Trains the model for a given number of epochs, with early stopping.
            evaluate(): Evaluates the model on the test data.
            predict(inputs): Makes predictions on a batch of inputs.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.model.to(device)
        
    def train(self, epochs, patience=3):
        train_losses = []
        train_maes = []
        val_losses = []
        val_maes = []
        patience_counter = 0
        maes_max = 5e10
        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            running_mae = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                mae = torch.abs(outputs - labels).mean()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_mae = running_mae / len(self.train_loader.dataset)
            train_losses.append(epoch_loss)
            train_maes.append(epoch_mae)
            
            # Validation
            self.model.eval()
            running_val_loss = 0.0
            running_val_mae = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    mae = torch.abs(outputs - labels).mean()
                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_mae += mae.item() * inputs.size(0)
            epoch_val_loss = running_val_loss / len(self.val_loader.dataset)
            epoch_val_mae = running_val_mae / len(self.val_loader.dataset)
            val_losses.append(epoch_val_loss)
            val_maes.append(epoch_val_mae)
            self.scheduler.step(epoch_val_loss)
            print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train MAE: {epoch_mae:.4f} | Val Loss: {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.4f}")
            # Early stopping
            if epoch_val_mae < maes_max:
                maes_max = epoch_val_mae
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement after {} epochs".format(patience))
                break
        return train_losses, train_maes, val_losses, val_maes
    
    def evaluate(self):
        self.model.eval()
        total_mae = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                mae = torch.abs(outputs - labels).mean()
                total_mae += mae.item() * inputs.size(0)
                total_samples += inputs.size(0)
        test_mae = total_mae / total_samples
        print(f'Test MAE: {test_mae:.4f}')
        return test_mae
    
    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
        return preds
    