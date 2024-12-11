from torch import nn
import torch
from transformers import Swinv2Model, Swinv2Config

class ImageEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_dim[0], 16, 9, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            flatten_size = self.main(torch.zeros(1, *input_dim)).shape[1]
        self.linear = nn.Linear(flatten_size, output_dim)
    
    def forward(self, x):
        x = self.main(x)
        return self.linear(x)
    
class ImageEncoderSWIN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageEncoderSWIN, self).__init__()
        config = Swinv2Config(
            image_size=input_dim[1],
            num_channels=input_dim[0],
            embed_dim=48,
        )
        self.main = Swinv2Model(config)
        self.linear = nn.Linear(config.hidden_size, output_dim)
    
    def forward(self, x):
        x = self.main(x)
        return self.linear(x.pooler_output)