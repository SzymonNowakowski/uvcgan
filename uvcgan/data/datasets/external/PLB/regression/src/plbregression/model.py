import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TimmWrapper(torch.nn.Module):
    def __init__(self, backbone_name, pretrained, in_chans, num_classes, preprocessor=None, postprocessor=None):
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def forward(self, x):
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        x = self.model.forward(x)
        if self.postprocessor is not None:
            x = self.postprocessor(x)
        return x

class SimpleRegressionNet(nn.Module):
    def __init__(self, input_channels=1, num_outputs=5, preprocessor=None, postprocessor=None):
        super(SimpleRegressionNet, self).__init__()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # After 5 max pools of a 224x224 image:
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        if self.postprocessor is not None:
            x = self.postprocessor(x)
        return x


class SimpleAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, kq_dim=64, use_bias=False):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(dim_in, kq_dim))
        self.to_k = torch.nn.Linear(dim_in, kq_dim, bias=use_bias)
        self.to_v = torch.nn.Linear(dim_in, dim_out, bias=use_bias)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dim_out = dim_out
    def forward(self, x):
        # x: (batch_size, dim_in, h, w)
        batch_size, dim_in, h, w = x.shape
        x_flat = x.view(batch_size, dim_in, h * w).transpose(1, 2) # (batch_size, h*w, dim_in)
        q = self.q.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, dim_in, kq_dim)
        k = self.to_k(x_flat) # (batch_size, h*w, kq_dim)
        v = self.to_v(x_flat) # (batch_size, h*w, dim_out)
        attn_scores = torch.bmm(q.transpose(1, 2), k.transpose(1, 2)) / (k.shape[-1] ** 0.5) # (batch_size, dim_in, h*w)
        attn_weights = self.softmax(attn_scores) # (batch_size, dim_in, h*w)
        attn_output = torch.bmm(attn_weights, v) # (batch_size, dim_in, dim_out)
        attn_output = attn_output.transpose(1, 2).view(batch_size, self.dim_out, h, w) # (batch_size, dim_out, h, w)
        return attn_output