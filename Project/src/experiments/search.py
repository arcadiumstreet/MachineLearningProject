from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import numpy as np

from src.training.evaluator import TorchEvaluator
from src.utils.utils import sample_nested_config
from src.models.model_registry import MODEL_REGISTRY
from src.training.trainer import Trainer
from src.data.data_loader import TimeSeriesDataset



def search(config, datasets, trainer_library):
    match trainer_library:
        case "sklearn":
            best_model = search_sklearn(config, datasets)
        case "torch":
            best_model = search_torch(config, datasets)
        case _:
            raise ValueError(f"Trainer library {trainer_library} not supported")
    
    return best_model

def search_torch(config, datasets):
    n_trials = config["search"]["n_trials"]
    search_space = config["search_space"]


    # ONLY for TIME SERIES DATASETS
    (X_train, y_train), (X_test, y_test) = datasets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    lookback =  54
    lookahead = 1
    
    train_dataset = TimeSeriesDataset(X_train, lookback, lookahead, target=y_train)
    valid_dataset = TimeSeriesDataset(X_valid, lookback, lookahead, target=y_valid)

    train_dataloader = train_dataset.to_dataloader(shuffle=False)
    valid_dataloader = valid_dataset.to_dataloader(shuffle=False)
    datasets = (train_dataloader, valid_dataloader)
    
    evaluator = TorchEvaluator(valid_dataloader, device="cpu")
    

    for i in range(n_trials):

        # sample a configuration - recursively sample while preserving structure
        model_cfg = sample_nested_config(search_space["model"])
        training_cfg = sample_nested_config(search_space["training"])

        # create model
        model = MODEL_REGISTRY[config["model"]](model_cfg)
        
        # train
        # Convert optimizer string to actual optimizer class
        optimizer_class = getattr(optim, training_cfg["optimizer"])
        optimizer = optimizer_class(model.parameters(), **training_cfg["optimizer_args"] if training_cfg["optimizer_args"] else {})
        
        scheduler_class = getattr(optim.lr_scheduler, training_cfg["lr_scheduler"])
        scheduler = scheduler_class(optimizer, **training_cfg["lr_scheduler_args"] if training_cfg["lr_scheduler_args"] else {})

        trainer = Trainer(model=model, 
                        data_loader=datasets, 
                        device="cpu",
                        optimizer=optimizer, 
                        scheduler=scheduler,
                        criterion=BCEWithLogitsLoss())

        trainer.train(epochs=config["epochs"])

        # evaluate
        metrics = evaluator.evaluate(model=trainer.get_trained_model(), 
                            cfg={"model": model_cfg, "training": training_cfg})
        print("-------------------------------- \n")

    return evaluator.best_model_params()


def search_sklearn(config, datasets):
    n_trials = config["search"]["n_trials"]
    model = MODEL_REGISTRY[config["model"]](config["search_space"]) 
    model_cfg = config["search_space"]

    # distributions = dict(C=uniform(loc=0, scale=4),
    #                     penalty=['l2', 'l1'])

    (X_train, y_train), (X_test, y_test) = datasets # return 
    # Defensive sanitization to avoid native-level crashes in some sklearn estimators
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).ravel()

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float32).ravel()

    tscv = TimeSeriesSplit(n_splits=3)
  
    clf = RandomizedSearchCV(
        model,
        model_cfg,
        cv=tscv,
        n_iter=n_trials
    )
    search = clf.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_



