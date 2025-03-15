import torch
from torch import nn


class SLPNet(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_classes: int,
                 model_dropout: float,
                 ):
        super().__init__()

        self.downsample = nn.Conv1d(
            in_channels=1024,
            out_channels=model_dim,
            kernel_size=1,
            bias=False
        )

        self.dropout = nn.Dropout(p=model_dropout)
        self.batchnorm = nn.BatchNorm1d(model_dim, affine=False)

        self.temporal_layer = nn.LSTM(input_size=model_dim,
                                      hidden_size=model_dim)

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(
            in_features=model_dim, out_features=num_classes)

    def forward(self, x):
        x_c = self.downsample(x)
        x_c = self.batchnorm(self.relu(x_c))
        x_c = self.dropout(x_c)

        x_c = torch.permute(x_c, (0,2,1))
        
        x_c, _ = self.temporal_layer(x_c)
        x_c = self.relu(x_c)

        x_c = self.classifier(x_c).squeeze()
        return x_c
