import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from .text_attention_net import TextAttentionNet
from typing import Union
from collections import OrderedDict
from sklearn.model_selection import train_test_split



class HybridStockSystem:
    def __init__(self, numerical_model=None, text_model=None, meta_learner=None):
        # Branch A: The Number Cruncher
        self.numerical_model = numerical_model
        self.is_num_model_trained = False if numerical_model is None else True
        if numerical_model is None:
            self.numerical_model = RandomForestClassifier(n_estimators=100, 
                                                    max_depth=10)
        
        # Branch B: The News Reader
        self.text_model = text_model
        self.is_text_model_trained = False if text_model is None else True
        if text_model is None:
            self.text_model = TextAttentionNet()
        elif isinstance(text_model, (dict, OrderedDict)):
            # If a state dict is provided, create a new model and load the state dict
            self.text_model = TextAttentionNet()
            self.text_model.load_state_dict(text_model)
            self.is_text_model_trained = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model.to(self.device)
        
        # The Merger: Learns how much to trust each branch
        self.meta_learner = meta_learner
        self.is_meta_learner_trained = False if meta_learner is None else True
        if meta_learner is None:
            self.meta_learner = LogisticRegression()
        
    

    def fit(self, X_tech, X_text, y):
        """
        X_tech: 2D Numpy Array (Samples, Features)
        X_text: 3D Numpy Array (Samples, Headlines, Embeddings)
        y: Target Array
        """
        # 1. Train Random Forest (Branch A)
        print("Training Branch A (Numerical Model) on Technicals")
        if not self.is_num_model_trained:
            self.numerical_model.fit(X_tech, y)
            self.is_num_model_trained = True
        
        # 2. Train Neural Net (Branch B)
        print("Training Branch B (Text Model) on News")
        if not self.is_text_model_trained:
            self._train_pytorch(X_text, y)
            self.is_text_model_trained = True
        
        # 3. Train Meta-Learner (The Merger)
        print("Training Meta-Learner (Fusion)")
        
        # Get predictions from A (on training data - ideally use cross-val here to prevent leak)
        prob_num = self.numerical_model.predict_proba(X_tech)[:, 1] # Probability of Class 1
        # Get predictions from B
        self.text_model.eval()
        with torch.no_grad():
            tensor_text = torch.tensor(X_text, dtype=torch.float32).to(self.device)
            prob_text = self.text_model(tensor_text).cpu().numpy().flatten()
            
        # Stack them: Shape (N, 2) -> [[prob_rf, prob_text], ...]
        X_meta = np.column_stack((prob_num, prob_text))
        
        if not self.is_meta_learner_trained:
            self.meta_learner.fit(X_meta, y)
            self.is_meta_learner_trained = True
        print("System Trained.")
        print(f"  ->  Fusion Weights: {self.meta_learner.coef_[0][0]:.2f}, Text: {self.meta_learner.coef_[0][1]:.2f}")



    # trainer for torch
    def _train_pytorch(self, text_data, labels, epochs=100, batch_size=32, patience=10):
        """Internal helper to train the neural net loop"""

        X_train, X_val, y_train, y_val = train_test_split(text_data, labels, test_size=0.2, shuffle=False)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True)
        
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        val_loader = DataLoader(val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False)
        
        optimizer = optim.Adam(self.text_model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', 
                                                        factor=0.5,
                                                        patience=patience)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            total_loss = 0
            self.text_model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.text_model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
            epoch_train_loss = total_loss / len(train_loader.dataset)

            # validation
            self.text_model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.text_model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
            epoch_val_loss = running_val_loss / len(val_loader.dataset)

            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

            # Step the learning rate scheduler
            scheduler.step(epoch_val_loss)

            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
                    break

    def predict(self, X_tech, X_text):
        # 1. Get RF Opinion
        prob_num = self.numerical_model.predict_proba(X_tech)[:, 1]
        
        # 2. Get Text Opinion
        self.text_model.eval()
        with torch.no_grad():
            if isinstance(X_text, Union[np.ndarray, list]):
                X_text = torch.tensor(X_text, dtype=torch.float32).to(self.device)
            else:
                X_text = X_text.to(self.device)
            prob_text = self.text_model(X_text).cpu().numpy().flatten()
            
        # 3. Merge
        X_meta = np.column_stack((prob_num, prob_text))
        final_pred = self.meta_learner.predict_proba(X_meta)
        return final_pred





