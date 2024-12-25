import torch.nn as nn


class CNN_MNIST(nn.Module):
    def __init__(self, 
                 CHANNELS,
                 OUTPUT_BREADTH,
                 HIDDEN_BREADTH,
                 BIAS, DROPOUT_P):
        super().__init__()

        dropout = nn.Dropout(p=DROPOUT_P)
        activation = nn.ReLU()
        a, b, c = CHANNELS

        self.architecture = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=a,
                      kernel_size=3,
                      stride=1,
                      padding=1),           # (*, 8, 8, 1) -> (*, 8, 8, a)
            dropout,
            activation,
            nn.Conv2d(in_channels=a,
                      out_channels=b,
                      kernel_size=3,
                      stride=1,
                      padding=1),           # (*, 8, 8, 16) -> (*, 8, 8, b)
            nn.MaxPool2d(kernel_size=2),    # (*, 8, 8, b) -> (*, 4, 4, b)
            dropout,
            activation,
            nn.Conv2d(in_channels=b,
                      out_channels=c,
                      kernel_size=3,
                      stride=1,
                      padding=1),           # (*, 4, 4, 64) -> (*, 4, 4, c)
            dropout,
            activation,
            nn.Flatten(),
            
            # 2 FULLY-CONNECTED LAYERS
            nn.Linear(in_features=4*4*c, 
                    out_features=HIDDEN_BREADTH,
                    bias=BIAS),
            dropout,
            activation,
            nn.Linear(in_features=HIDDEN_BREADTH, 
                    out_features=OUTPUT_BREADTH,
                    bias=BIAS),
        )
    
    def forward(self, X):
        # shape needs to be (B, C, NX, NY), ie. channel second!
        # hence .unsqueeze(1)
        return self.architecture(X.unsqueeze(1))