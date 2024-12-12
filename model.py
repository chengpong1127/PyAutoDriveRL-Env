from torch import nn
import torch
from transformers import AutoModel
import torchvision.transforms as T

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_type="avg"):
        super(ConvBlock, self).__init__()
        pool_layer = nn.AvgPool2d(2) if pool_type == "avg" else nn.MaxPool2d(2)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            pool_layer
        )

    def forward(self, x):
        return self.block(x)

class ImageEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, pool_type="avg"):
        super(ImageEncoder, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(input_dim[0], 64, kernel_size=9, stride=2, padding=1, pool_type=pool_type),
            ConvBlock(64, 64, kernel_size=5, stride=2, padding=1, pool_type=pool_type),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, pool_type=pool_type),
            nn.Flatten(),
        )
        with torch.no_grad():
            flatten_size = self.main(torch.zeros(1, *input_dim)).shape[1]
        self.linear = nn.Linear(flatten_size, output_dim)
        
        self.gaussian_blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        self.noise_level = 0.05 
    
    def forward(self, x):
        #x += torch.randn_like(x) * self.noise_level
        x = self.gaussian_blur(x)
        x = self.main(x)
        return self.linear(x)
    
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