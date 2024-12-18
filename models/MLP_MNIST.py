import torch.nn as nn


class MLP_MNIST(nn.Module):
    def __init__(self, 
                 INPUT_BREADTH, OUTPUT_BREADTH,
                 HIDDEN_DEPTH, HIDDEN_BREADTH,
                 BIAS, DROPOUT_P):
        super().__init__()

        dropout = nn.Dropout(p=DROPOUT_P)
        activation = nn.ReLU()

        self.architecture = nn.Sequential(
            nn.Flatten(start_dim=1),
            # First Layer
            nn.Linear(in_features=INPUT_BREADTH, 
                      out_features=HIDDEN_BREADTH,
                      bias=BIAS),
            dropout,
            activation,

            # Hidden Layers
            *(HIDDEN_DEPTH * [
                nn.Linear(in_features=HIDDEN_BREADTH, 
                      out_features=HIDDEN_BREADTH,
                      bias=BIAS),
                dropout, 
                activation]
            ),
            
            # Final Layer
            nn.Linear(in_features=HIDDEN_BREADTH, 
                      out_features=OUTPUT_BREADTH,
                      bias=BIAS)
        )
    
    def forward(self, X):
        return self.architecture(X)