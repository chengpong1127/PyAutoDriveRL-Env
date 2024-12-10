from torch import nn
import torch
from transformers import AutoModel

class ImageEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 2, 1),
            nn.ReLU(),
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
        self.swin = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        with torch.no_grad():
            flatten_size = self.swin(torch.zeros(1, *input_dim)).pooler_output.shape[1]
        self.linear = nn.Linear(flatten_size, output_dim)
    
    def forward(self, x):
        with torch.no_grad():
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
            x = self.swin(x).pooler_output
        return self.linear(x)