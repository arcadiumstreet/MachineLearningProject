from sklearn.linear_model import ElasticNet

def create_elastic_net(config):
    return ElasticNet(**config)