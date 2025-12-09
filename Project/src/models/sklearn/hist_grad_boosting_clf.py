from sklearn.ensemble import GradientBoostingClassifier

def create_grad_boosting_clf(config):
    return GradientBoostingClassifier(**config)