from torch import nn


class SmartSequential(nn.Module):
    module_dict = {}

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
        "A": lambda x: SmartSequential.activation_functions[x],
        "F": lambda x: nn.Flatten(start_dim=x),
        "M": lambda x: nn.MaxPool2d(kernel_size=x),
        "D": lambda x: nn.Dropout(p=x),
    }

    def __init__(self):
        super().__init__()

        self.architecture = nn.Sequential(
            *self.get_architecture_layers()
        )
    
    @staticmethod
    def extend_layers(layers, layer_config):
        for l in layer_config:
            if isinstance(l[0], int):
                for _ in range(l[0]):
                    layers = SmartSequential.extend_layers(layers, l[1])
                continue

            layers.append(SmartSequential.config2layer[l[0]](l[1]))
        return layers

    def get_architecture_layers(self):        
        layers = SmartSequential.extend_layers(layers=[], layer_config=self.CONFIG)
        return layers
    
    def state_dict(self):
        sd = super().state_dict()
        sd["CONFIG"] = self.CONFIG
        return sd

    def load_state_dict(self, state_dict, strict=False):
        # override the default artument to strict=False, since SmartModules 
        # contain the extra key "CONFIG"
        return super().load_state_dict(state_dict, strict)