from torch import nn
import torch
from transformers import AutoModel
import torchvision.transforms as T

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_type="avg"):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ImageEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, pool_type="avg"):
        super(ImageEncoder, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(input_dim[0], 16, kernel_size=9, stride=2, padding=0, pool_type=pool_type),
            ConvBlock(16, 32, kernel_size=5, stride=2, padding=0, pool_type=pool_type),
            ConvBlock(32, 32, kernel_size=3, stride=2, padding=0, pool_type=pool_type),
            nn.Flatten(),
        )
        # with torch.no_grad():
        #     flatten_size = self.main(torch.zeros(1, *input_dim)).shape[1]
        # self.linear = nn.Linear(flatten_size, output_dim)
    
    def forward(self, x):
        return self.main(x)
    
class ImageEncoderSWIN(nn.Module):
    def __init__(self, output_dim):
        super(ImageEncoderSWIN, self).__init__()
        self.main = AutoModel.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.linear = nn.Linear(self.main.config.hidden_size, output_dim)
    
    def forward(self, x):
        if x.shape[-2:] != (256, 256):
            x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear')
        with torch.no_grad():
            x = self.main(x)
        return self.linear(x.pooler_output)