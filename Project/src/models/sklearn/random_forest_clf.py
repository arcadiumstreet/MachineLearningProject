from sklearn.ensemble import RandomForestClassifier

def create_random_forest_clf(config):
    return RandomForestClassifier(**config)
