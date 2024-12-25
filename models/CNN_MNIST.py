from models.SmartSequential import SmartSequential


class CNN_MNIST(SmartSequential):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        super().__init__()
    
    
    def forward(self, X):
        # shape needs to be (B, C, NX, NY), ie. channel second!
        # hence .unsqueeze(1)
        return self.architecture(X.unsqueeze(1))

SmartSequential.module_dict["CNN_MNIST"] = CNN_MNIST