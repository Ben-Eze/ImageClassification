from models.SmartSequential import SmartSequential


class MLP_MNIST(SmartSequential):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        super().__init__()
    
    
    def forward(self, X):
        return self.architecture(X)

SmartSequential.module_dict["MLP_MNIST"] = MLP_MNIST
