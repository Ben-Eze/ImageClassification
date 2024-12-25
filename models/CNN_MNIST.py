import torch.nn as nn


class CNN_MNIST(nn.Module):
    activation_functions = {
         "relu": nn.ReLU(),
         "tanh": nn.Tanh(),
    }

    # M: nn.MaxPool2d()
    # F: nn.Flatten()
    # C: nn.Conv2d()
    # L: nn.Linear()
    # A: activation
    # D: nn.Dropout

    config2layer = {
        "L": lambda x: nn.Linear(in_features=x[0], 
                                 out_features=x[1]),
        "C": lambda x: nn.Conv2d(in_channels=x[0],
                                 out_channels=x[1],
                                 kernel_size=3,
                                 stride=1,
                                 padding=1),
        "A": lambda x: CNN_MNIST.activation_functions[x],
        "F": lambda _: nn.Flatten(),
        "M": lambda x: nn.MaxPool2d(kernel_size=x),
        "D": lambda x: nn.Dropout(p=x),
    }
    def __init__(self, CONFIG):
        super().__init__()

        
        self.CONFIG = CONFIG

        self.architecture = nn.Sequential(
            *self.get_architecture_layers()
        )
    
    def get_architecture_layers(self):        
        layers = []
        for l in self.CONFIG:
            layers.append(CNN_MNIST.config2layer[l[0]](l[1]))
        return layers
    
    def forward(self, X):
        # shape needs to be (B, C, NX, NY), ie. channel second!
        # hence .unsqueeze(1)
        return self.architecture(X.unsqueeze(1))
    
    def state_dict(self):
        sd = super().state_dict()
        sd["CONFIG"] = self.CONFIG
        return sd

    def load_state_dict(self, state_dict, strict=False):
        # override the default artument to strict=False, since SmartModules 
        # contain the exxtra key "CONFIG"
        return super().load_state_dict(state_dict, strict)