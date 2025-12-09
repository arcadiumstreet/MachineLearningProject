import torch
from torch.nn import Sigmoid
import numpy as np

class Trainer:
    def __init__(self, model, data_loader, criterion, 
                    optimizer, device, scheduler=None) -> None:
        """
        A utility class for training, validating and testing a PyTorch model.

        Attributes:
            model (nn.Module): The PyTorch model to train.
            data_loader (DataLoader): The DataLoader for the training and validation data.
            criterion (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer.
            device (torch.device): The device to train on (e.g., 'cpu' or 'cuda' or 'mps').
            scheduler (Scheduler): The scheduler.

        Methods:
            train(epochs, patience): Trains the model for a given number of epochs, with early stopping.

        """
        self.model = model
        self.train_loader = data_loader[0]
        self.val_loader = data_loader[1]
        # self.test_loader = data_loader[2]
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
            if self.scheduler is not None:
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
    
    def get_trained_model(self):
        return self.model

    # def evaluate(self):
    #     self.model.eval()
    #     total_accuracy = 0.0
    #     total_samples = 0
    #     with torch.no_grad():
    #         for inputs, labels in self.test_loader:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             outputs = torch.sigmoid(self.model(inputs))
    #             outputs_binary = (outputs >= 0.5).float()
    #             # Ensure shapes match by squeezing/reshaping if needed
    #             outputs_binary = outputs_binary.squeeze().cpu().numpy()
    #             labels = labels.squeeze().cpu().numpy()
    #             # Convert boolean comparison to float before taking mean
    #             accuracy = np.mean(outputs_binary == labels)
    #             total_accuracy += accuracy * inputs.size(0)
    #             total_samples += inputs.size(0)
    #     total_accuracy = total_accuracy / total_samples
    #     print(f'Test Accuracy: {total_accuracy:.4f}')
    #     return total_accuracy
    