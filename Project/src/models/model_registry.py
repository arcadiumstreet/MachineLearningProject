from .sklearn.random_forest_clf import create_random_forest_clf
from .sklearn.hist_grad_boosting_clf import create_grad_boosting_clf
from .torch.lstm import create_lstm
from .torch.rnn import create_rnn
from .sklearn.elastic_net import create_elastic_net
from src.training.trainer import Trainer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

MODEL_REGISTRY = {
    "RandomForestClassifier": create_random_forest_clf,
    "LSTM": create_lstm,
    "RNN": create_rnn,
    "ElasticNet": create_elastic_net,
    "GradientBoostingClassifier": create_grad_boosting_clf,
}

TRAINER_REGISTRY = {
    "Trainer": Trainer,
    "RandomizedSearchCV": RandomizedSearchCV,
    "GridSearchCV": GridSearchCV,
}