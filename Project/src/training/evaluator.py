from curses import A_CHARTEXT
import torch   
from sklearn.base import BaseEstimator

from sklearn.metrics import recall_score, roc_curve, \
                            precision_score, f1_score, \
                            accuracy_score, roc_auc_score, \
                            ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def plot_roc_curve(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

def plot_confusion_matrix(self, y_true, y_pred):
    plt.rc('font', size=9)  # extra code – make the text smaller
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()
    
class TorchEvaluator:
    def __init__(self, test_loader, device="cpu"):
        self.test_loader = test_loader
        self.device = device

        self.best_models = None
        self.best_metrics = None
        self.best_params = None
        self.metric_of_interest = accuracy_score

    # ?????
    # 2 class -> one ouput -> sigmoid 
    # 2 class -> n outputs -> softmax

    def best_model_params(self):
        return self.best_models, self.best_params

    def evaluate(self, model, cfg):
        model.eval()

        y_true = []
        y_pred = []
        y_pred_proba = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                outputs_binary = (outputs >= 0.5).astype(float)
                # Ensure shapes match by squeezing/reshaping if needed
                labels = labels.squeeze().cpu().numpy()

                y_true.extend(labels)
                y_pred.extend(outputs_binary)
                y_pred_proba.extend(outputs)


        if self.best_models is None:
            self.best_metrics = self.metric_of_interest(y_true, y_pred)
            self.best_models = model
            self.best_params = cfg
            return self.best_metrics
        else:
            current_metrics = self.metric_of_interest(y_true, y_pred)
            if current_metrics > self.best_metrics:
                self.best_metrics = current_metrics
                self.best_models = model
                self.best_params = cfg
            
            return current_metrics
        

class FinalEvaluator:
    _registry_predict = {}
    _registry_predict_proba = {}

    def __init__(self, test_loader, lib, device="cpu"):
        self.test_loader = test_loader
        self.lib = lib
        self.device = device

    @staticmethod
    def get_all_metrics(y_true, y_pred):
        return pd.DataFrame({
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_true, y_pred), 4)
        }, index=[0])

    # ---- Register predict ----
    @classmethod
    def register_predict(cls, model_cls):
        def decorator(func):
            cls._registry_predict[model_cls] = func
            return func
        return decorator

    def predict(self, model, x):
        for cls, fn in self._registry_predict.items():
            if isinstance(model, cls):
                return fn(self, model, x)
        raise ValueError(f"No registered predict() for {type(model)}")


    @classmethod
    def register_predict_proba(cls, model_cls):
        def decorator(func):
            cls._registry_predict_proba[model_cls] = func
            return func
        return decorator

    def predict_proba(self, model, x):
        for cls, fn in self._registry_predict_proba.items():
            if isinstance(model, cls):
                return fn(self, model, x)
        raise ValueError(f"No registered predict_proba() for {type(model)}")



# ---- Register sklearn ----
@FinalEvaluator.register_predict(BaseEstimator)
def _(self, model, x):
    return model.predict(x)

@FinalEvaluator.register_predict_proba(BaseEstimator)
def _(self, model, x):
    return model.predict_proba(x)


# ---- Register torch ----
@FinalEvaluator.register_predict(torch.nn.Module)
def _(self, model, x):
    model.eval()
    with torch.no_grad():
        # unsqueeze to have (batch_size, 1, input_size) (for LSTM)
        # TODO : make it more general
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) 
        inputs = x.to(self.device)
        outputs = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
        return (outputs >= 0.5).astype(float)

@FinalEvaluator.register_predict_proba(torch.nn.Module)
def _(self, model, x):
    model.eval()
    with torch.no_grad():
        # unsqueeze to have (batch_size, 1, input_size) (for LSTM)
        # TODO : make it more general
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        inputs = x.to(device=self.device)
        outputs = model(inputs)
        return torch.sigmoid(outputs).squeeze().cpu().numpy()